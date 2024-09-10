#include "motis/module/dispatcher.h"

#include <queue>
#include <string_view>

#include "boost/asio/post.hpp"
#include "boost/system/system_error.hpp"

#include "fmt/format.h"

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

#include "ctx/ctx.h"

#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/error.h"
#include "motis/module/global_res_ids.h"
#include "motis/module/module.h"
#include "version.h"

namespace motis::module {

dispatcher* dispatcher::direct_mode_dispatcher_ = nullptr;  // NOLINT

dispatcher::dispatcher(registry& reg,
                       std::vector<std::unique_ptr<module>> modules)
    : ctx::access_scheduler<ctx_data>(
          to_res_id(global_res_id::FIRST_FREE_RES_ID)),
      registry_{reg},
      modules_{std::move(modules)} {}

void dispatcher::register_timer(char const* name,
                                boost::posix_time::time_duration interval,
                                std::function<void()>&& fn,
                                ctx::accesses_t&& access) {
  auto const inserted =
      timers_
          .emplace(name,
                   std::make_shared<timer>(name, this, interval, std::move(fn),
                                           std::move(access)))
          .second;
  utl::verify(inserted, "register_timer: {} already registered", name);
}

void dispatcher::start_timers() {
  for (auto const& [name, t] : timers_) {
    t->exec(boost::system::error_code{});
  }
}

void dispatcher::on_msg(msg_ptr const& msg, callback const& cb) {
  dispatch(msg, cb, ctx::op_id("dispatcher::on_msg"), ctx::op_type_t::IO);
}

void dispatcher::on_connect(std::string const& target, client_hdl const& c) {
  auto const it = registry_.client_handlers_.find(target);
  if (it != end(registry_.client_handlers_)) {
    it->second(c);
  } else {
    LOG(logging::warn) << "no ws handler found for target \"" << target << "\"";
  }
}

bool dispatcher::connect_ok(std::string const& target) {
  return registry_.client_handlers_.find(target) !=
         end(registry_.client_handlers_);
}

std::vector<future> dispatcher::publish(msg_ptr const& msg,
                                        ctx_data const& data, ctx::op_id id) {
  id.name = msg->get()->destination()->target()->str();
  auto it = registry_.topic_subscriptions_.find(id.name);
  if (it == end(registry_.topic_subscriptions_)) {
    return {};
  }

  return utl::to_vec(it->second, [&](auto&& op) {
    if (direct_mode_dispatcher_ != nullptr) {
      auto f = std::make_shared<ctx::future<ctx_data, msg_ptr>>(id);
      f->set(op.fn_(msg));
      return f;
    } else {
      return post_work(data, [&, msg] { return op.fn_(msg); }, id);
    }
  });
}

future dispatcher::req(msg_ptr const& msg, ctx_data const& data,
                       ctx::op_id const& id) {
  auto f = make_future(id);
  dispatch(
      msg,
      [f](msg_ptr res, std::error_code ec) mutable {
        if (ec) {
          f->set(std::make_exception_ptr(std::system_error{ec}));
        } else if (res && res->get()->content_type() == MsgContent_MotisError) {
          f->set(
              std::make_exception_ptr(std::system_error{error::remote_error}));
        } else {
          f->set(std::move(res));
        }
      },
      id, ctx::op_type_t::WORK, &data);
  return f;
}

void dispatcher::dispatch(msg_ptr const& msg, callback const& cb, ctx::op_id id,
                          ctx::op_type_t const op_type, ctx_data const* data) {
  id.name = msg->get()->destination()->target()->str();

  auto op_name = registry_.get_operation_name(id.name);
  if (id.name == "/api") {
    op_name = id.name;
  }
  auto span = motis_tracer->StartSpan(
      op_name.value_or(std::string_view{"unknown target"}),
      {{"ctx.op_id.name", id.name},
       {"ctx.op_id.index", id.index},
       {"ctx.op_id.created_at", id.created_at},
       {"ctx.op_id.parent_index", id.parent_index}});
  auto scope = opentelemetry::trace::Scope{span};

  if (id.name == "/api") {
    return cb(api_desc(msg->id()), std::error_code{});
  }

  auto const run = [this, id, cb, msg, span]() {
    try {
      if (auto const op = registry_.get_operation(id.name)) {
        return cb(op->fn_(msg), std::error_code());
      } else if (auto const remote_op = registry_.get_remote_op(id.name);
                 remote_op.has_value()) {
        boost::asio::post(runner_.ios_,
                          [op = remote_op.value(), msg, cb]() { op(msg, cb); });
        return;
      } else {
        LOG(logging::warn) << "target not found: " << id.name;
        span->SetStatus(opentelemetry::trace::StatusCode::kError,
                        "target not found");
        return handle_no_target(msg, cb);
      }
    } catch (std::system_error const& e) {
      span->AddEvent("exception",
                     {{"exception.message", e.what()},
                      {"exception.error.code", e.code().value()},
                      {"exception.error.category", e.code().category().name()},
                      {"exception.error.message", e.code().message()}});
      span->SetStatus(opentelemetry::trace::StatusCode::kError,
                      "system error exception");
      return cb(nullptr, e.code());
    } catch (std::exception const& e) {
      span->AddEvent("exception", {
                                      {"exception.message", e.what()},
                                  });
      span->SetStatus(opentelemetry::trace::StatusCode::kError, "exception");
      LOG(logging::error) << "error executing " << id.name << ": " << e.what();
      return cb(nullptr, error::unknown_error);
    } catch (...) {
      span->AddEvent("exception", {{"exception.type", "unknown"}});
      span->SetStatus(opentelemetry::trace::StatusCode::kError,
                      "unknown error");
      LOG(logging::error) << "unknown error executing " << id.name;
      return cb(nullptr, error::unknown_error);
    }
  };

  if (direct_mode_dispatcher_ != nullptr) {
    run();
  } else {
    auto access = ctx::accesses_t{};
    if (auto const op = registry_.get_operation(id.name); op) {
      access = op->access_;
    }

    auto new_data = data != nullptr ? ctx_data{*data} : ctx_data{this};
    new_data.otel_context_stack_.push_back(
        opentelemetry::context::RuntimeContext::GetCurrent());
    enqueue(
        std::move(new_data), [run]() { run(); }, id, op_type,
        std::move(access));
  }
}

msg_ptr dispatcher::api_desc(int const id) const {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_ApiDescription,
      CreateApiDescription(
          fbb, fbb.CreateString(MOTIS_GIT_TAG),
          fbb.CreateVector(utl::to_vec(
              registry_.operations_,
              [&](auto&& op) { return fbb.CreateString(op.first); })))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(fbb);
}

void dispatcher::handle_no_target(msg_ptr const& msg, callback const& cb) {
  if (queue_no_target_msgs_) {
    no_target_msg_queue_.emplace(msg, cb);
  } else {
    return cb(nullptr, error::target_not_found);
  }
}

void dispatcher::retry_no_target_msgs() {
  while (!no_target_msg_queue_.empty()) {
    auto const [msg, cb] = no_target_msg_queue_.front();
    no_target_msg_queue_.pop();
    on_msg(msg, cb);
  }
}

}  // namespace motis::module
