#include "motis/module/timer.h"

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

#include "motis/core/common/logging.h"
#include "motis/core/otel/tracer.h"
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

  auto span =
      motis_tracer->StartSpan(name_, {{"timer.interval", interval_.seconds()}});
  auto scope = opentelemetry::trace::Scope{span};

  auto access_copy = access_;
  auto data = ctx_data{dispatcher_};
  data.otel_context_stack_.push_back(
      opentelemetry::context::RuntimeContext::GetCurrent());
  dispatcher_->enqueue(
      std::move(data),
      [self = shared_from_this(), span]() {
        try {
          self->fn_();
        } catch (std::exception const& e) {
          span->AddEvent("exception", {
                                          {"exception.message", e.what()},
                                      });
          span->SetStatus(opentelemetry::trace::StatusCode::kError,
                          "exception");
          LOG(logging::error)
              << "error in timer " << self->name_ << ": " << e.what();
        } catch (...) {
          span->AddEvent("exception", {{"exception.type", "unknown"}});
          span->SetStatus(opentelemetry::trace::StatusCode::kError,
                          "unknown error");
          LOG(logging::error) << "unknown error in timer " << self->name_;
        }
      },
      ctx::op_id{name_, CTX_LOCATION, 0U}, ctx::op_type_t::IO,
      std::move(access_copy));

  schedule();
}

}  // namespace motis::module
