#pragma once

#include <fstream>
#include <streambuf>

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"
#include "motis/module/progress_listener.h"

namespace motis::module {

struct clog_redirect : public std::streambuf {
  clog_redirect(progress_listener&, std::string name,
                char const* log_file_path);

  clog_redirect(clog_redirect const&) = delete;
  clog_redirect(clog_redirect&&) = delete;
  clog_redirect& operator=(clog_redirect const&) = delete;
  clog_redirect& operator=(clog_redirect&&) = delete;

  ~clog_redirect() override;

  int_type overflow(int_type) override;

  static void set_enabled(bool);

private:
  enum class output_state {
    NORMAL,
    MODE_SELECT,
    PERCENT,
    ERR,
    STATUS,
    BOUND_OUTPUT_LOW,
    BOUND_OUTPUT_HIGH,
    BOUND_INPUT_HIGH
  } state_{output_state::NORMAL};
  float output_low_{0.F}, output_high_{0.F};
  std::ofstream sink_;
  std::string name_;
  std::streambuf* backup_clog_;
  std::string buf_;
  progress_listener& progress_listener_;
  static bool enabled_;
};

}  // namespace motis::module
