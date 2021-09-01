#include "motis/module/dispatcher.h"

#include <queue>
#include <string_view>

#include "boost/asio/post.hpp"
#include "boost/system/system_error.hpp"

#include "ctx/ctx.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/module/error.h"

namespace motis::module {

dispatcher* dispatcher::direct_mode_dispatcher_ = nullptr;  // NOLINT

dispatcher::dispatcher(registry& reg,
                       std::vector<std::unique_ptr<module>>&& modules)
    : registry_{reg}, modules_{std::move(modules)} {}

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
    utl::verify(ctx::current_op<ctx_data>() == nullptr ||
                    ctx::current_op<ctx_data>()->data_.access_ >= op.access_,
                "match the access permissions of parent or be root operation");
    if (direct_mode_dispatcher_ != nullptr) {
      auto f = std::make_shared<ctx::future<ctx_data, msg_ptr>>(id);
      f->set(op.fn_(msg));
      return f;
    } else {
      return post_work(
          data, [&, msg] { return op.fn_(msg); }, id);
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

ctx::access_t dispatcher::access_of(std::string const& target) {
  if (auto const it = registry_.topic_subscriptions_.find(target);
      it != end(registry_.topic_subscriptions_)) {
    utl::verify(!registry_.topic_subscriptions_.empty(),
                "empty topic subscriptions");
    return std::max_element(
               begin(it->second), end(it->second),
               [](auto const& a, auto const& b) {
                 return static_cast<std::underlying_type_t<ctx::access_t>>(
                            a.access_) <
                        static_cast<std::underlying_type_t<ctx::access_t>>(
                            b.access_);
               })
        ->access_;
  } else if (auto const it = registry_.operations_.find(target);
             it != end(registry_.operations_)) {
    return it->second.access_;
  } else {
    return ctx::access_t::READ;
  }
}

void dispatcher::dispatch(msg_ptr const& msg, callback const& cb, ctx::op_id id,
                          ctx::op_type_t const op_type, ctx_data const* data) {
  id.name = msg->get()->destination()->target()->str();
  if (id.name == "/api") {
    return cb(api_desc(msg->id()), std::error_code{});
  }

  auto const run = [this, id, cb, msg]() {
    try {
      if (auto const op = registry_.get_operation(id.name)) {
        utl::verify(
            ctx::current_op<ctx_data>() == nullptr ||
                ctx::current_op<ctx_data>()->data_.access_ >= op->access_,
            "match the access permissions of parent or be root "
            "operation");
        return cb(op->fn_(msg), std::error_code());
      } else if (auto const remote_op = registry_.get_remote_op(id.name);
                 remote_op.has_value()) {
        boost::asio::post(runner_.ios_,
                          [op = remote_op.value(), msg, cb]() { op(msg, cb); });
        return;
      } else {
        return handle_no_target(msg, cb);
      }
    } catch (std::system_error const& e) {
      return cb(nullptr, e.code());
    } catch (std::exception const& e) {
      LOG(logging::error) << "error executing " << id.name << ": " << e.what();
      return cb(nullptr, error::unknown_error);
    } catch (...) {
      LOG(logging::error) << "unknown error executing " << id.name;
      return cb(nullptr, error::unknown_error);
    }
  };

  if (direct_mode_dispatcher_ != nullptr) {
    run();
  } else {
    ctx::access_t access{ctx::access_t::NONE};
    if (data != nullptr) {
      access = ctx::access_t::NONE;
    } else if (auto const op = registry_.get_operation(id.name); op) {
      access = op->access_;
    }

    enqueue(
        data != nullptr ? *data : ctx_data{access, this, &shared_data_},
        [run]() { run(); }, id, op_type, access);
  }
}

msg_ptr dispatcher::api_desc(int const id) const {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_ApiDescription,
      CreateApiDescription(
          fbb, fbb.CreateVector(utl::to_vec(
                   registry_.operations_,
                   [&](auto&& op) { return fbb.CreateString(op.first); })))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(fbb);
}

ctx::access_t dispatcher::access_of(msg_ptr const& msg) {
  return access_of(msg->get()->destination()->target()->str());
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
