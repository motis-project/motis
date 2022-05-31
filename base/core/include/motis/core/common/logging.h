#pragma once

#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>

#include "fmt/format.h"

#ifdef _MSC_VER
#define MOTIS_GMT(a, b) gmtime_s(b, a)
#else
#define MOTIS_GMT(a, b) gmtime_r(a, b)
#endif

#define FILE_NAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG(lvl)                                                      \
  motis::logging::log() << "[" << motis::logging::str[lvl] << "]"     \
                        << "[" << motis::logging::time() << "]"       \
                        << "[" << FILE_NAME << ":" << __LINE__ << "]" \
                        << " "

namespace motis::logging {

std::string time(time_t);
std::string time();

struct log {
  log() : lock_{log_mutex_} {}

  log(log const&) = delete;
  log& operator=(log const&) = delete;

  log(log&&) = default;
  log& operator=(log&&) = default;

  template <typename T>
  friend log&& operator<<(log&& l, T&& t) {
    std::clog << std::forward<T&&>(t);
    return std::move(l);
  }

  ~log() { std::clog << std::endl; }

  std::unique_lock<std::mutex> lock_;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::mutex log_mutex_;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool enabled_;
};

enum log_level { emrg, alrt, crit, error, warn, notice, info, debug };

static const char* const str[]{"emrg", "alrt", "crit", "erro",
                               "warn", "note", "info", "debg"};

template <typename Msg, typename... Args>
void l(log_level const lvl, Msg&& msg, Args&&... args) {
  motis::logging::log() << "[" << motis::logging::str[lvl] << "]"
                        << "[" << motis::logging::time() << "]"
                        << "[" << FILE_NAME << ":" << __LINE__ << "]"
                        << " "
                        << fmt::format(std::forward<Msg>(msg),
                                       std::forward<Args>(args)...);
}

struct scoped_timer final {
  explicit scoped_timer(std::string name);
  scoped_timer(scoped_timer const&) = delete;
  scoped_timer(scoped_timer&&) = delete;
  scoped_timer& operator=(scoped_timer const&) = delete;
  scoped_timer& operator=(scoped_timer&&) = delete;
  ~scoped_timer();

  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

struct manual_timer final {
  explicit manual_timer(std::string name);
  void stop_and_print();
  double duration_ms() const;

  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> stop_;
};

}  // namespace motis::logging
