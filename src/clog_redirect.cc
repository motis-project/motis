#include "motis/clog_redirect.h"

#include <iostream>
#include <mutex>

namespace motis {

namespace {

struct synchronized_streambuf : std::streambuf {
  synchronized_streambuf(std::streambuf* wrapped, std::mutex& mutex)
      : wrapped_{wrapped}, mutex_{mutex} {}

  int_type overflow(int_type ch) override {
    auto const lock = std::lock_guard{mutex_};
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return wrapped_->pubsync() == 0 ? traits_type::not_eof(ch)
                                      : traits_type::eof();
    }
    return wrapped_->sputc(traits_type::to_char_type(ch));
  }

  std::streamsize xsputn(char const* s, std::streamsize count) override {
    auto const lock = std::lock_guard{mutex_};
    return wrapped_->sputn(s, count);
  }

  int sync() override {
    auto const lock = std::lock_guard{mutex_};
    return wrapped_->pubsync();
  }

private:
  std::streambuf* wrapped_;
  std::mutex& mutex_;
};

}  // namespace

clog_redirect::clog_redirect(char const* log_file_path) {
  if (!enabled_) {
    return;
  }

  sink_.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  sink_.open(log_file_path, std::ios_base::app);
  sink_buf_ = std::make_unique<synchronized_streambuf>(sink_.rdbuf(), mutex_);

  backup_clog_ = std::clog.rdbuf();
  std::clog.rdbuf(sink_buf_.get());
  active_ = true;
}

clog_redirect::~clog_redirect() {
  if (!active_) {
    return;
  }

  auto const lock = std::lock_guard{mutex_};
  std::clog.rdbuf(backup_clog_);
}

void clog_redirect::set_enabled(bool const enabled) { enabled_ = enabled; }

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool clog_redirect::enabled_ = true;

}  // namespace motis