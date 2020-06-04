#include "motis/module/clog_redirect.h"

#include <iostream>

namespace motis::module {

clog_redirect::clog_redirect(char const* log_file_path)
    : backup_clog_{std::clog.rdbuf()} {
  if (!enabled_) {
    return;
  }

  sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  sink_.open(log_file_path, std::ios_base::app);
  std::clog.rdbuf(sink_.rdbuf());
}

clog_redirect::~clog_redirect() {
  if (enabled_) {
    std::clog.rdbuf(backup_clog_);
  }
}

void clog_redirect::set_enabled(bool const enabled) { enabled_ = enabled; }

bool clog_redirect::enabled_ = true;

}  // namespace motis::module
