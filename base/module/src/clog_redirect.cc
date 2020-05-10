#include "motis/module/clog_redirect.h"

#if defined(ERROR)
#undef ERROR
#endif

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

void update_error(std::string const& name, std::string const& error) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(name),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
          motis::import::Status::Status_ERROR, fbb.CreateString(error), 0)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
}

void update_progress(std::string const& name, int progress) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(name),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
          motis::import::Status::Status_RUNNING, fbb.CreateString(""), progress)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
}

clog_redirect::clog_redirect(std::string name, char const* log_file_path)
    : name_{std::move(name)}, backup_clog_{std::clog.rdbuf()} {
  sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  sink_.open(log_file_path, std::ios_base::app);
  std::clog.rdbuf(this);
}

clog_redirect::~clog_redirect() { std::clog.rdbuf(backup_clog_); }

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
        state_ = output_state::ERROR;
      } else {
        state_ = output_state::PERCENT;
        percent_ = traits_type::to_char_type(c) - '0';
      }
      return c;

    case output_state::PERCENT:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        update_progress(name_, percent_);
        state_ = output_state::NORMAL;
      } else {
        percent_ *= 10;
        percent_ += traits_type::to_char_type(c) - '0';
      }
      return c;

    case output_state::ERROR:
      if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
        update_error(name_, error_);
        error_.clear();
        state_ = output_state::NORMAL;
      } else {
        error_ += traits_type::to_char_type(c);
      }
      return c;

    default: return c;
  }
}

}  // namespace motis::module
