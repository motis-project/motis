#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <streambuf>

namespace motis {

struct clog_redirect {
  explicit clog_redirect(char const* log_file_path);

  clog_redirect(clog_redirect const&) = delete;
  clog_redirect(clog_redirect&&) = delete;
  clog_redirect& operator=(clog_redirect const&) = delete;
  clog_redirect& operator=(clog_redirect&&) = delete;

  ~clog_redirect();

  static void set_enabled(bool);

private:
  std::ofstream sink_;
  std::unique_ptr<std::streambuf> sink_buf_;
  std::streambuf* backup_clog_{};
  bool active_{};
  std::mutex mutex_;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool enabled_;
};

}  // namespace motis
