#pragma once

#include <fstream>
#include <streambuf>

namespace motis::module {

struct clog_redirect : public std::streambuf {
  clog_redirect(std::string name, char const* log_file_path);

  clog_redirect(clog_redirect const&) = delete;
  clog_redirect(clog_redirect&&) = delete;

  clog_redirect& operator=(clog_redirect const&) = delete;
  clog_redirect& operator=(clog_redirect&&) = delete;

  ~clog_redirect() override;

  int_type overflow(int_type c = traits_type::eof()) override;  // NOLINT

  static void disable();

private:
  enum class output_state {
    NORMAL,
    MODE_SELECT,
    PERCENT,
    ERR
  } state_{output_state::NORMAL};
  int percent_{0};
  std::ofstream sink_;
  std::string name_;
  std::streambuf* backup_clog_;
  std::string error_;
  static bool disabled_;
};

}  // namespace motis::module
