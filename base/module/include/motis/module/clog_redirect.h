#pragma once

#include <fstream>
#include <streambuf>

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

namespace motis::module {

struct log_streambuf : public std::streambuf {
  explicit log_streambuf(std::string name, char const* log_file_path)
      : name_{std::move(name)} {
    sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    sink_.open(log_file_path);

    backup_ = std::clog.rdbuf();
    std::clog.rdbuf(this);
  }

  log_streambuf(log_streambuf const&) = delete;
  log_streambuf(log_streambuf&&) = delete;

  log_streambuf& operator=(log_streambuf const&) = delete;
  log_streambuf& operator=(log_streambuf&&) = delete;

  ~log_streambuf() override { std::clog.rdbuf(backup_); }

  virtual int_type overflow(int_type c = traits_type::eof()) {
    static constexpr auto const PROGRESS_MARKER = '\0';
    if (traits_type::to_char_type(c) == PROGRESS_MARKER) {
      percent_active_ = !percent_active_;
      if (!percent_active_) {
        update_status();
        percent_ = 0;
      }
      return c;
    } else if (percent_active_) {
      percent_ *= 10;
      percent_ += traits_type::to_char_type(c) - '0';
      return c;
    } else {
      return sink_.rdbuf()->sputc(c);
    }
  }

  void update_status() const {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_StatusUpdate,
        motis::import::CreateStatusUpdate(
            fbb, fbb.CreateString(name_),
            fbb.CreateVector(
                std::vector<flatbuffers::Offset<flatbuffers::String>>{}),
            motis::import::Status::Status_RUNNING, percent_)
            .Union(),
        "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
  }

  bool percent_active_{false};
  int percent_{0};
  std::ofstream sink_;
  std::string name_;
  std::streambuf* backup_;
};

}  // namespace motis::module