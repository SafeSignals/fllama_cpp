// fllama_decision.cpp — llama-server's decision API (/v1/systemone) with no
// HTTP server.
//
// llama-server answers a decision request in server_routes::post_systemone
// (req). Here the request body comes from Dart, and the response body goes
// back to Dart's callback, unchanged. See fllama.h.

#include "fllama.h"
#include "fllama_internal.h"

#include "server-common.h"
#include "server-context.h"
#include "server-http.h"

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// The batch of a decision model when the request sets no context size. The
// decision models that llama.cpp serves read the prompt both ways, so the
// whole prompt, with its images and audio, must fit in one ubatch.
constexpr int kDefaultDecisionContext = 4096;

// The strings of a request, owned: Dart may free its copies when
// fllama_decision() returns.
struct owned_request {
  fllama_inference_request request;
  std::string model_path;
  std::string mmproj_path;
  std::string gpu_device_key;
  std::string body;

  owned_request(const fllama_inference_request &r, const char *body_json)
      : request(r), body(body_json ? body_json : "") {
    auto copy = [](const char *src, std::string &dst) -> char * {
      if (src == nullptr) return nullptr;
      dst = src;
      return dst.data();
    };
    request.model_path = copy(r.model_path, model_path);
    request.model_mmproj_path = copy(r.model_mmproj_path, mmproj_path);
    request.gpu_device_key = copy(r.gpu_device_key, gpu_device_key);
    // Not used by a decision.
    request.input = nullptr;
    request.openai_request_json_string = nullptr;
    request.draft_model_path = nullptr;
  }

  owned_request(const owned_request &) = delete;
  owned_request &operator=(const owned_request &) = delete;
};

// The result of a request: the response body, or an error.
struct decision_outcome {
  std::string response;
  std::string error_type;
  std::string error_message;

  void fail(std::string type, std::string message) {
    error_type = std::move(type);
    error_message = std::move(message);
  }
};

// fllama's error type of an error response of llama-server.
std::string error_type_of(const json &error, const std::string &message) {
  if (message.find("Compute error") != std::string::npos ||
      message.find("failed to decode") != std::string::npos ||
      message.find("out of memory") != std::string::npos ||
      message.find("OutOfMemory") != std::string::npos) {
    return "backend";
  }
  const std::string type = error.value("type", "");
  if (type == "exceed_context_size_error") return "exceed_context_size";
  // The prompt does not fit in the batch, which is the context size.
  if (message.find("too large to process") != std::string::npos) {
    return "exceed_context_size";
  }
  if (type == "invalid_request_error") return "invalid_request";
  if (type == "not_supported_error") return "not_supported";
  return "other";
}

void run_body(const owned_request &owned, request_log_scope &llama_logs,
              decision_outcome &outcome) {
  const fllama_inference_request &request = owned.request;
  const int rid = request.request_id;
  ServerManager &mgr = fllama_server_manager();
  const int64_t t0 = ggml_time_ms();

  fllama_backends_init_once();
  common_params params = fllama_load_params(request);
  if (params.n_ctx <= 0) params.n_ctx = kDefaultDecisionContext;
  // The whole prompt in one ubatch: see kDefaultDecisionContext.
  params.n_batch = params.n_ctx;
  params.n_ubatch = params.n_ctx;

  ServerResources *srv = mgr.get_or_create(
      owned.model_path, params, request.dart_logger,
      [rid, &mgr] { return mgr.is_cancelled(rid); });
  if (!srv || !srv->srv_ctx || !srv->routes) {
    if (mgr.is_cancelled(rid)) {
      outcome.fail("cancelled", "Cancelled");
      return;
    }
    std::string message = "Failed to load model " + owned.model_path;
    if (!llama_logs.first_error().empty()) {
      message += ": " + llama_logs.first_error();
    }
    outcome.fail("model_load", message);
    return;
  }
  struct release_guard {
    ServerManager &mgr;
    const std::string &path;
    ~release_guard() { mgr.release(path); }
  } guard{mgr, owned.model_path};

  if (mgr.is_cancelled(rid)) {
    outcome.fail("cancelled", "Cancelled");
    return;
  }

  fllama_log("[fllama] Decision: model ready (" +
                 std::to_string(ggml_time_ms() - t0) + " ms)",
             request.dart_logger);

  const std::function<bool()> should_stop = [rid, &mgr] {
    return mgr.is_cancelled(rid);
  };
  server_http_req req{{}, {}, "/v1/systemone", "", owned.body, {},
                      should_stop};
  server_http_res_ptr res;
  try {
    res = srv->routes->post_systemone(req);
  } catch (const std::invalid_argument &e) {
    outcome.fail("invalid_request", e.what());
    return;
  } catch (const common_json_error &e) {
    outcome.fail("invalid_request", e.what());
    return;
  }
  if (mgr.is_cancelled(rid)) {
    outcome.fail("cancelled", "Cancelled");
    return;
  }
  if (!res || res->data.empty()) {
    outcome.fail("other", "No response");
    return;
  }
  if (res->status != 200) {
    std::string message = res->data;
    json error = json::object();
    try {
      error = json::parse(res->data).value("error", json::object());
      message = error.value("message", res->data);
    } catch (const std::exception &) {
      // Not JSON: the text is the message.
    }
    const std::string type = error_type_of(error, message);
    if (type == "backend") mgr.mark_unhealthy(owned.model_path);
    outcome.fail(type, message);
    return;
  }
  outcome.response = std::move(res->data);
  fllama_log("[fllama] Decision done (" +
                 std::to_string(ggml_time_ms() - t0) + " ms)",
             request.dart_logger);
}

void run_decision(std::shared_ptr<owned_request> owned,
                  fllama_decision_callback callback) {
  const fllama_inference_request &request = owned->request;
  decision_outcome outcome;
  {
    request_log_scope llama_logs(request.dart_logger);
    try {
      run_body(*owned, llama_logs, outcome);
    } catch (const std::exception &e) {
      outcome.fail("other", e.what());
    } catch (...) {
      outcome.fail("other", "Unknown exception");
    }
    if (!outcome.error_type.empty() && outcome.error_type != "cancelled") {
      fllama_log("[fllama] Decision failed (" + outcome.error_type +
                     "): " + outcome.error_message,
                 request.dart_logger);
    }
    llama_logs.end_line();
  }
  fllama_server_manager().clear_cancel(request.request_id);
  fllama_server_manager().unregister_request_thread();
  if (outcome.error_type.empty()) {
    callback(request.request_id, fllama_dart_string(outcome.response),
             nullptr);
  } else {
    json error = {{"type", outcome.error_type},
                  {"message", outcome.error_message}};
    callback(request.request_id, nullptr,
             fllama_dart_string(safe_json_to_str(error)));
  }
}

} // namespace

extern "C" {

FFI_PLUGIN_EXPORT int fllama_decision(struct fllama_inference_request request,
                                      const char *body_json,
                                      fllama_decision_callback callback) {
  if (request.model_path == nullptr || request.model_path[0] == '\0' ||
      body_json == nullptr || callback == nullptr) {
    return 0;
  }
  auto owned = std::make_shared<owned_request>(request, body_json);
  fllama_server_manager().start_request_thread(
      [owned, callback] { run_decision(owned, callback); });
  return 1;
}

} // extern "C"
