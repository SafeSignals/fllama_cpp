// fllama_realtime.cpp — llama-server's realtime API (/v1/realtime) with no
// HTTP server.
//
// llama-server runs a realtime session in server_routes::ws_realtime(req,
// conn), on one thread for the life of a WebSocket. Here conn is a
// [bridge_conn]: its read() takes the client events that Dart sends with
// fllama_realtime_send(), and its send() gives the server events to Dart's
// callback. The events are the JSON text of the WebSocket messages,
// unchanged. See fllama.h and telosnex's ADR 015.

#include "fllama.h"
#include "fllama_internal.h"

#include "server-common.h"
#include "server-context.h"
#include "server-http.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// One session's WebSocket. read() runs on llama-server's reader thread;
// send() on its threads and on the session's request thread.
class bridge_conn final : public server_http_context::ws_conn {
public:
  bridge_conn(int session, fllama_realtime_event_callback callback)
      : session_(session), callback_(callback) {}

  // Blocks until a client event comes or the connection closes.
  int read(std::string &msg) override {
    std::unique_lock<std::mutex> lk(mutex_);
    inbox_cv_.wait(lk, [this] { return !inbox_.empty() || !open_; });
    if (!open_) return 0;
    msg = std::move(inbox_.front());
    inbox_.pop_front();
    return 1;
  }

  // Drops events after the connection closes: llama-server sends its last
  // error before it closes, and Dart wants none after its close().
  bool send(const std::string &msg) override {
    std::lock_guard<std::mutex> lk(send_mutex_);
    if (finished_ || !callback_ || !is_open()) return false;
    callback_(session_, fllama_dart_string(msg));
    return true;
  }

  // The realtime API sends text only.
  bool send_binary(const std::string &) override { return false; }

  bool is_open() const override {
    std::lock_guard<std::mutex> lk(mutex_);
    return open_;
  }

  // The server ends the session.
  void close() override { close_with(false); }

  // Dart ends the session.
  void close_by_client() { close_with(true); }

  bool closed_by_client() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return closed_by_client_;
  }

  // A client event. False if the connection is closed.
  bool push(std::string msg) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (!open_) return false;
      inbox_.push_back(std::move(msg));
    }
    inbox_cv_.notify_one();
    return true;
  }

  // Sends the last event. After it, send() calls nothing. Call once, after
  // the session's threads end.
  void finish(const std::string &reason) {
    std::lock_guard<std::mutex> lk(send_mutex_);
    finished_ = true;
    if (!callback_) return;
    json ev = {{"type", "fllama.realtime.closed"}, {"reason", reason}};
    callback_(session_, fllama_dart_string(safe_json_to_str(ev)));
  }

private:
  void close_with(bool by_client) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (!open_) return;
      open_ = false;
      closed_by_client_ = by_client;
      inbox_.clear();
    }
    inbox_cv_.notify_all();
  }

  const int session_;
  const fllama_realtime_event_callback callback_;

  mutable std::mutex mutex_;
  std::condition_variable inbox_cv_;
  std::deque<std::string> inbox_;
  bool open_ = true;
  bool closed_by_client_ = false;

  // Orders the callbacks, and keeps every one before the last.
  std::mutex send_mutex_;
  bool finished_ = false;
};

std::mutex g_sessions_lock;
std::unordered_map<int, std::shared_ptr<bridge_conn>> g_sessions;
std::atomic<int> g_next_session{1};

std::shared_ptr<bridge_conn> find_session(int session) {
  std::lock_guard<std::mutex> lk(g_sessions_lock);
  auto it = g_sessions.find(session);
  return it == g_sessions.end() ? nullptr : it->second;
}

// The strings of a request, owned: Dart may free its copies when
// fllama_realtime_open() returns.
struct owned_request {
  fllama_inference_request request;
  std::string model_path;
  std::string mmproj_path;
  std::string draft_model_path;
  std::string gpu_device_key;

  explicit owned_request(const fllama_inference_request &r) : request(r) {
    auto copy = [](const char *src, std::string &dst) -> char * {
      if (src == nullptr) return nullptr;
      dst = src;
      return dst.data();
    };
    request.model_path = copy(r.model_path, model_path);
    request.model_mmproj_path = copy(r.model_mmproj_path, mmproj_path);
    request.draft_model_path = copy(r.draft_model_path, draft_model_path);
    request.gpu_device_key = copy(r.gpu_device_key, gpu_device_key);
    // Not used by a realtime session.
    request.input = nullptr;
    request.openai_request_json_string = nullptr;
  }

  owned_request(const owned_request &) = delete;
  owned_request &operator=(const owned_request &) = delete;
};

// "a=1&b=2" as a map. No percent-decoding.
std::map<std::string, std::string> parse_query(const std::string &query) {
  std::map<std::string, std::string> params;
  size_t start = 0;
  while (start <= query.size()) {
    size_t end = query.find('&', start);
    if (end == std::string::npos) end = query.size();
    const std::string pair = query.substr(start, end - start);
    if (!pair.empty()) {
      const size_t eq = pair.find('=');
      if (eq == std::string::npos) {
        params[pair] = "";
      } else {
        params[pair.substr(0, eq)] = pair.substr(eq + 1);
      }
    }
    start = end + 1;
  }
  return params;
}

void send_error(bridge_conn &conn, const std::string &type,
                const std::string &message) {
  json ev = {{"type", "error"},
             {"event_id", "event_" + random_string()},
             {"error",
              {{"type", type},
               {"code", nullptr},
               {"message", message},
               {"param", nullptr}}}};
  conn.send(safe_json_to_str(ev));
}

// Loads the model, then runs the session until it ends. Returns the reason
// of the closed event.
std::string run_session(const owned_request &owned, const std::string &query,
                        bridge_conn &conn, request_log_scope &llama_logs) {
  const fllama_inference_request &request = owned.request;
  ServerManager &mgr = fllama_server_manager();
  const int64_t t0 = ggml_time_ms();

  if (!conn.is_open()) return "client";
  fllama_backends_init_once();
  common_params params = fllama_load_params(request);
  // A realtime request has no max_tokens. 0 as the global n_predict would
  // stop a live transcription after its first token: the stream task of
  // Voxtral Realtime sets no limit of its own. Responses of a conversation
  // set theirs (max_output_tokens).
  if (params.n_predict <= 0) params.n_predict = -1;
  ServerResources *srv = mgr.get_or_create(
      owned.model_path, params, request.dart_logger,
      [&conn] { return !conn.is_open(); });
  if (!srv || !srv->srv_ctx || !srv->routes) {
    if (conn.closed_by_client()) return "client";
    std::string message = "Failed to load model " + owned.model_path;
    if (!llama_logs.first_error().empty()) {
      message += ": " + llama_logs.first_error();
    }
    send_error(conn, "model_load", message);
    return "error";
  }
  struct release_guard {
    ServerManager &mgr;
    const std::string &path;
    ~release_guard() { mgr.release(path); }
  } guard{mgr, owned.model_path};
  // A close during the load.
  if (!conn.is_open()) return "client";

  fllama_log("[fllama] Realtime session: model ready (" +
                 std::to_string(ggml_time_ms() - t0) + " ms)",
             request.dart_logger);

  const std::function<bool()> should_stop = [&conn] {
    return !conn.is_open();
  };
  server_http_req req{parse_query(query), {}, "/v1/realtime", query, "", {},
                      should_stop};
  srv->routes->ws_realtime(req, conn);
  return conn.closed_by_client() ? "client" : "server";
}

void run_realtime(int session, std::shared_ptr<owned_request> owned,
                  std::string query, std::shared_ptr<bridge_conn> conn) {
  const fllama_log_callback logger = owned->request.dart_logger;
  std::string reason;
  {
    request_log_scope llama_logs(logger);
    try {
      reason = run_session(*owned, query, *conn, llama_logs);
    } catch (const std::exception &e) {
      send_error(*conn, "server_error", e.what());
      reason = "error";
    } catch (...) {
      send_error(*conn, "server_error", "Unknown exception");
      reason = "error";
    }
    fllama_log("[fllama] Realtime session ended: " + reason, logger);
    llama_logs.end_line();
  }
  conn->close();
  {
    std::lock_guard<std::mutex> lk(g_sessions_lock);
    g_sessions.erase(session);
  }
  fllama_server_manager().unregister_request_thread();
  conn->finish(reason);
}

} // namespace

extern "C" {

FFI_PLUGIN_EXPORT int
fllama_realtime_open(struct fllama_inference_request request,
                     const char *query,
                     fllama_realtime_event_callback on_event) {
  if (request.model_path == nullptr || request.model_path[0] == '\0' ||
      on_event == nullptr) {
    return 0;
  }
  const int session = g_next_session.fetch_add(1);
  auto owned = std::make_shared<owned_request>(request);
  auto conn = std::make_shared<bridge_conn>(session, on_event);
  std::string query_copy = query ? query : "";
  {
    std::lock_guard<std::mutex> lk(g_sessions_lock);
    g_sessions.emplace(session, conn);
  }
  fllama_server_manager().start_request_thread(
      [session, owned, query_copy, conn] {
        run_realtime(session, owned, query_copy, conn);
      });
  return session;
}

FFI_PLUGIN_EXPORT int fllama_realtime_send(int session,
                                           const char *event_json) {
  if (event_json == nullptr) return 0;
  auto conn = find_session(session);
  return conn && conn->push(event_json) ? 1 : 0;
}

FFI_PLUGIN_EXPORT void fllama_realtime_close(int session) {
  if (auto conn = find_session(session)) conn->close_by_client();
}

} // extern "C"
