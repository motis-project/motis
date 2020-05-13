#include "motis/module/clog_redirect.h"

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

clog_redirect::clog_redirect(std::string name, char const* log_file_path)
    : name_{std::move(name)},
      backup_clog_{std::clog.rdbuf()},
      dispatcher_{ctx::current_op<ctx_data>()->data_.dispatcher_},
      op_id_{ctx::current_op<ctx_data>()->id_.index},
      op_data_{ctx::current_op<ctx_data>()->data_} {
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

void clog_redirect::publish(msg_ptr const& msg) {
  auto id = ctx::op_id(CTX_LOCATION);
  id.parent_index = op_id_;
  ctx::await_all(dispatcher_->publish(msg, op_data_, id));
}

void clog_redirect::update_error(std::string const& error) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(name_),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
          motis::import::Status::Status_ERROR, fbb.CreateString(error), 0)
          .Union(),
      "/import", DestinationType_Topic);
  publish(make_msg(fbb));
}

void clog_redirect::update_progress(int const progress) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(name_),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
          motis::import::Status::Status_RUNNING, fbb.CreateString(""), progress)
          .Union(),
      "/import", DestinationType_Topic);
  publish(make_msg(fbb));
}

void clog_redirect::disable() { disabled_ = true; }

bool clog_redirect::disabled_ = false;

}  // namespace motis::module
