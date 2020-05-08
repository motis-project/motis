#pragma once

#if defined(ERROR)
#undef ERROR
#endif

#include <fstream>
#include <streambuf>

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

struct log_streambuf : public std::streambuf {
  explicit log_streambuf(std::string name, char const* log_file_path)
      : name_{std::move(name)}, backup_clog_{std::clog.rdbuf()} {
    sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    sink_.open(log_file_path);
    std::clog.rdbuf(this);
  }

  log_streambuf(log_streambuf const&) = delete;
  log_streambuf(log_streambuf&&) = delete;

  log_streambuf& operator=(log_streambuf const&) = delete;
  log_streambuf& operator=(log_streambuf&&) = delete;

  ~log_streambuf() override { std::clog.rdbuf(backup_clog_); }

  int_type overflow(int_type c = traits_type::eof()) override {  // NOLINT
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
          update_progress();
          state_ = output_state::NORMAL;
        } else {
          percent_ *= 10;
          percent_ += traits_type::to_char_type(c) - '0';
        }
        return c;

      case output_state::ERROR:
        if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
          update_error();
          error_.clear();
          state_ = output_state::NORMAL;
        } else {
          error_ += traits_type::to_char_type(c);
        }
        return c;

      default: return c;
    }
  }

private:
  void update_error() const {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_StatusUpdate,
        motis::import::CreateStatusUpdate(
            fbb, fbb.CreateString(name_),
            fbb.CreateVector(
                std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
            motis::import::Status::Status_ERROR, fbb.CreateString(error_), 0)
            .Union(),
        "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
  }

  void update_progress() const {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_StatusUpdate,
        motis::import::CreateStatusUpdate(
            fbb, fbb.CreateString(name_),
            fbb.CreateVector(
                std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
            motis::import::Status::Status_RUNNING, fbb.CreateString(""),
            percent_)
            .Union(),
        "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
  }

  enum class output_state {
    NORMAL,
    MODE_SELECT,
    PERCENT,
    ERROR
  } state_{output_state::NORMAL};
  int percent_{0};
  std::ofstream sink_;
  std::string name_;
  std::streambuf* backup_clog_;
  std::string error_;
};

}  // namespace motis::module