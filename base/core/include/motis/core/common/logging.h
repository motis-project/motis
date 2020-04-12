#pragma once

#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>

#ifdef _MSC_VER
#define gmt(a, b) gmtime_s(b, a)
#else
#define gmt(a, b) gmtime_r(a, b)
#endif

#define FILE_NAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG(lvl)                                                      \
  motis::logging::log() << "[" << motis::logging::str[lvl] << "]"     \
                        << "[" << motis::logging::time() << "]"       \
                        << "[" << FILE_NAME << ":" << __LINE__ << "]" \
                        << " "

namespace motis::logging {

struct log {
  log() : lock_{log_mutex_} {}

  log(log const&) = delete;
  log& operator=(log const&) = delete;

  log(log&&) = default;
  log& operator=(log&&) = default;

  template <typename T>
  friend log&& operator<<(log&& l, T&& t) {
    std::cout << std::forward<T&&>(t);
    return std::move(l);
  }

  ~log() { std::cout << std::endl; }

  std::unique_lock<std::mutex> lock_;
  static std::mutex log_mutex_;
};

enum log_level { emrg, alrt, crit, error, warn, notice, info, debug };

static const char* const str[]{"emrg", "alrt", "crit", "erro",
                               "warn", "note", "info", "debg"};

inline std::string time(time_t const t) {
  char buf[sizeof "2011-10-08t07:07:09z-0430"];
  struct tm result {};
  gmt(&t, &result);
  strftime(static_cast<char*>(buf), sizeof buf, "%FT%TZ%z", &result);
  return buf;
}

inline std::string time() {
  time_t now = 0;
  std::time(&now);
  return time(now);
}

struct scoped_timer final {
  explicit scoped_timer(std::string name)
      : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {
    LOG(info) << "[" << name_ << "] starting";
  }

  scoped_timer(scoped_timer const&) = delete;
  scoped_timer(scoped_timer&&) = delete;
  scoped_timer& operator=(scoped_timer const&) = delete;
  scoped_timer& operator=(scoped_timer&&) = delete;

  ~scoped_timer() {
    using namespace std::chrono;
    auto stop = steady_clock::now();
    double t = duration_cast<microseconds>(stop - start_).count() / 1000.0;
    LOG(info) << "[" << name_ << "] finished"
              << " (" << t << "ms)";
  }

  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

struct manual_timer final {
  explicit manual_timer(std::string name)
      : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {
    LOG(info) << "[" << name_ << "] starting";
  }

  void stop_and_print() {
    using namespace std::chrono;
    auto stop = steady_clock::now();
    double t = duration_cast<microseconds>(stop - start_).count() / 1000.0;
    LOG(info) << "[" << name_ << "] finished"
              << " (" << t << "ms)";
  }

  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

}  // namespace motis::logging
