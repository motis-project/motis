#include "motis/module/timer.h"

#include "motis/core/common/logging.h"
#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"

namespace motis::module {

timer::timer(char const* name, dispatcher* d,
             boost::posix_time::time_duration interval,
             std::function<void()> fn, ctx::accesses_t&& access)
    : name_{name},
      interval_{interval},
      timer_{d->runner_.ios()},
      fn_{std::move(fn)},
      dispatcher_{d},
      access_{std::move(access)} {}

void timer::stop() {
  stopped_ = true;
  timer_.cancel();
}

void timer::schedule() {
  if (stopped_) {
    return;
  }

  timer_.expires_from_now(interval_);
  timer_.async_wait(
      [self = shared_from_this()](boost::system::error_code const& ec) {
        self->exec(ec);
      });
}

void timer::exec(boost::system::error_code const& ec) {
  using namespace logging;

  if (stopped_ || ec == boost::asio::error::operation_aborted) {
    return;
  }

  auto access_copy = access_;
  dispatcher_->enqueue(
      ctx_data{dispatcher_},
      [self = shared_from_this()]() {
        try {
          self->fn_();
        } catch (std::exception const& e) {
          LOG(logging::error)
              << "error in timer " << self->name_ << ": " << e.what();
        } catch (...) {
          LOG(logging::error) << "unknown error in timer " << self->name_;
        }
      },
      ctx::op_id{name_, CTX_LOCATION, 0U}, ctx::op_type_t::IO,
      std::move(access_copy));

  schedule();
}

}  // namespace motis::module
