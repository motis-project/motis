#include "motis/module/clog_redirect.h"

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

clog_redirect::clog_redirect(progress_listener& progress_listener,
                             std::string name, char const* log_file_path)
    : name_{std::move(name)},
      backup_clog_{std::clog.rdbuf()},
      progress_listener_{progress_listener} {
  if (disabled_) {
    return;
  }

  sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  sink_.open(log_file_path, std::ios_base::app);
  std::clog.rdbuf(this);
}

clog_redirect::~clog_redirect() {
  if (!disabled_) {
    std::clog.rdbuf(backup_clog_);
  }
}

clog_redirect::int_type clog_redirect::overflow(clog_redirect::int_type c) {
  static constexpr auto const PROGRESS_MARKER = '\0';
  switch (state_) {
    case output_state::NORMAL:
      if (c == PROGRESS_MARKER) {
        state_ = output_state::MODE_SELECT;
        return c;
      } else {
        auto const ret = sink_.rdbuf()->sputc(c);
        sink_ << std::flush;
        return ret;
      }

    case output_state::MODE_SELECT:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        state_ = output_state::NORMAL;
      } else if (traits_type::to_char_type(c) == 'E') {
        state_ = output_state::ERR;
      } else {
        state_ = output_state::PERCENT;
        percent_ = traits_type::to_char_type(c) - '0';
      }
      return c;

    case output_state::PERCENT:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        update_progress(percent_);
        state_ = output_state::NORMAL;
      } else {
        percent_ *= 10;
        percent_ += traits_type::to_char_type(c) - '0';
      }
      return c;

    case output_state::ERR:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        update_error(error_);
        error_.clear();
        state_ = output_state::NORMAL;
      } else {
        error_ += traits_type::to_char_type(c);
      }
      return c;

    default: return c;
  }
}

void clog_redirect::update_error(std::string const& error) {
  progress_listener_.report_error(name_, error);
}

void clog_redirect::update_progress(int const progress) {
  progress_listener_.update_progress(name_, progress);
}

void clog_redirect::disable() { disabled_ = true; }

bool clog_redirect::disabled_ = false;

}  // namespace motis::module
