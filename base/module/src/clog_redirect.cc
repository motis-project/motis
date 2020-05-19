#include "motis/module/clog_redirect.h"

#include <iostream>

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

clog_redirect::clog_redirect(progress_listener& progress_listener,
                             std::string name, char const* log_file_path)
    : name_{std::move(name)},
      backup_clog_{std::clog.rdbuf()},
      progress_listener_{progress_listener} {
  if (!enabled_) {
    return;
  }

  sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  sink_.open(log_file_path, std::ios_base::app);
  std::clog.rdbuf(this);
}

clog_redirect::~clog_redirect() {
  if (enabled_) {
    std::clog.rdbuf(backup_clog_);
  }
}

clog_redirect::int_type clog_redirect::overflow(clog_redirect::int_type c) {
  auto const consume_float = [this] {
    auto val = std::stof(buf_);
    buf_.clear();
    return val;
  };

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
      switch (traits_type::to_char_type(c)) {
        case PROGRESS_MARKER: state_ = output_state::NORMAL; break;
        case 'E': state_ = output_state::ERR; break;
        case 'S': state_ = output_state::STATUS; break;
        case 'B': state_ = output_state::BOUND_OUTPUT_LOW; break;
        default:
          state_ = output_state::PERCENT;
          buf_ += traits_type::to_char_type(c);
      }
      return c;

    case output_state::PERCENT:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        progress_listener_.update_progress(name_, consume_float());
        state_ = output_state::NORMAL;
      } else {
        buf_ += traits_type::to_char_type(c);
      }
      return c;

    case output_state::ERR:
    case output_state::STATUS:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        state_ == output_state::ERR
            ? progress_listener_.report_error(name_, buf_)
            : progress_listener_.report_step(name_, buf_);
        buf_.clear();
        state_ = output_state::NORMAL;
      } else {
        buf_ += traits_type::to_char_type(c);
      }
      return c;

    case output_state::BOUND_OUTPUT_LOW:
      if (traits_type::to_char_type(c) == ' ') {
        output_low_ = consume_float();
        state_ = output_state::BOUND_OUTPUT_HIGH;
      } else {
        buf_ += traits_type::to_char_type(c);
      }
      return c;
    case output_state::BOUND_OUTPUT_HIGH:
      switch (traits_type::to_char_type(c)) {
        case ' ':
          output_high_ = consume_float();
          state_ = output_state::BOUND_INPUT_HIGH;
          break;
        case PROGRESS_MARKER:
          progress_listener_.set_progress_bounds(name_, output_low_,
                                                 consume_float(), 100.f);
          break;
        default: buf_ += traits_type::to_char_type(c);
      }
      return c;
    case output_state::BOUND_INPUT_HIGH:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        progress_listener_.set_progress_bounds(name_, output_low_, output_high_,
                                               consume_float());
        state_ = output_state::NORMAL;
      } else {
        buf_ += traits_type::to_char_type(c);
      }
      return c;

    default: return c;
  }
}

void clog_redirect::set_enabled(bool const enabled) { enabled_ = enabled; }

bool clog_redirect::enabled_ = true;

}  // namespace motis::module
