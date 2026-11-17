// fllama_internal.h — shared by fllama.cpp and fllama_realtime.cpp. Not a
// public header.
#ifndef FLLAMA_INTERNAL_H
#define FLLAMA_INTERNAL_H

#include "fllama.h"
#include "fllama_backends.h"
#include "fllama_inference_queue.h"

#include "llama.cpp/common/common.h"
#include "llama.cpp/ggml/include/ggml.h"

#include <string>

// The process's model cache.
ServerManager &fllama_server_manager();

// A copy of [s] that Dart owns and frees (dart_alloc.h).
char *fllama_dart_string(const std::string &s);

// Sends [msg] to [logger], or to stderr if there is no logger.
void fllama_log(const std::string &msg, fllama_log_callback logger = nullptr);

// The sampling settings that a GGUF file recommends (general.sampling.*),
// on top of llama.cpp's defaults. llama-server starts every request from
// these.
common_params_sampling fllama_model_sampling(const llama_model *model);

// The load-time params of [request], as every request builds them: two
// requests with the same fields share one cached context. Logs GPU choices
// to the request's logger. Call fllama_backends_init_once() first.
common_params fllama_load_params(const fllama_inference_request &request);

// While it exists, sends llama.cpp's log lines from this thread to the
// request's logger, and keeps the first error line. A model load runs on the
// request's thread, so its reason for failing is here.
class request_log_scope {
public:
  explicit request_log_scope(fllama_log_callback logger)
      : logger_(logger),
        scope_([this](enum ggml_log_level level, const char *text) {
          add(level, text ? text : "");
          return false; // Also to stderr.
        }) {}

  ~request_log_scope() { end_line(); }

  // The first error line that llama.cpp logged, or "".
  const std::string &first_error() const { return first_error_; }

  // Sends the line that llama.cpp has not ended. Call it before the request's
  // last callback: no log may reach Dart after it.
  void end_line() {
    if (!line_.empty()) {
      send(line_);
      line_.clear();
    }
  }

private:
  void add(enum ggml_log_level level, const std::string &text) {
    if (level != GGML_LOG_LEVEL_CONT) level_ = level;
    line_ += text;
    for (size_t end; (end = line_.find('\n')) != std::string::npos;) {
      send(line_.substr(0, end));
      line_.erase(0, end + 1);
    }
  }

  void send(const std::string &line) {
    if (line.empty()) return;
    if (level_ == GGML_LOG_LEVEL_ERROR && first_error_.empty()) {
      first_error_ = line;
    }
    if (logger_ && level_ >= GGML_LOG_LEVEL_INFO) {
      logger_(fllama_dart_string("[llama.cpp] " + line));
    }
  }

  fllama_log_callback logger_;
  enum ggml_log_level level_ = GGML_LOG_LEVEL_INFO;
  std::string line_;
  std::string first_error_;
  fllama_thread_log_scope scope_; // Last: it uses the members above.
};

#endif // FLLAMA_INTERNAL_H
