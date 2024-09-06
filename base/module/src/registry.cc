#include "motis/module/registry.h"

#include "boost/algorithm/string/predicate.hpp"

#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/module/message.h"
#include "motis/module/receiver.h"

namespace motis::module {

void registry::register_op(std::string const& name, op_fn_t fn,
                           ctx::accesses_t&& access) {
  auto const call = [fn_rec = std::move(fn),
                     name](msg_ptr const& m) -> msg_ptr { return fn_rec(m); };
  auto const inserted =
      operations_.emplace(name, op{std::move(call), std::move(access)}).second;
  utl::verify(inserted, "register_op: target {} already registered", name);
}

void registry::register_client_handler(
    std::string const& target, std::function<void(client_hdl)>&& handler) {
  auto const inserted =
      client_handlers_.emplace(target, std::move(handler)).second;
  utl::verify(inserted, "client_handler: target {} already registered", target);
}

void registry::subscribe(std::string const& topic, op_fn_t fn,
                         ctx::accesses_t&& access) {
  topic_subscriptions_[topic].emplace_back(
      [fn_rec = std::move(fn), topic](msg_ptr const& m) -> msg_ptr {
        return fn_rec(m);
      },
      std::move(access));
}

void registry::subscribe(std::string const& topic, void_op_fn_t fn,
                         ctx::accesses_t&& access) {
  subscribe(
      topic,
      [fn_rec = std::move(fn)](msg_ptr const&) {
        fn_rec();
        return msg_ptr{};
      },
      std::move(access));
}

std::vector<std::string> registry::register_remote_ops(
    std::vector<std::string> const& names, remote_op_fn_t const& fn) {
  std::lock_guard const g{remote_op_mutex_};
  std::vector<std::string> successful_names;
  for (auto const& name : names) {
    if (remote_operations_.emplace(name, fn).second) {
      successful_names.emplace_back(name);
    }
  }
  return successful_names;
}

void registry::unregister_remote_op(std::vector<std::string> const& names) {
  std::lock_guard const g{remote_op_mutex_};
  for (auto const& name : names) {
    remote_operations_.erase(name);
  }
}

std::optional<remote_op_fn_t> registry::get_remote_op(
    std::string const& prefix) {
  std::lock_guard const g{remote_op_mutex_};
  if (auto const it = remote_operations_.upper_bound(prefix);
      it != begin(remote_operations_) &&
      boost::algorithm::starts_with(prefix, std::next(it, -1)->first)) {
    return std::next(it, -1)->second;
  } else {
    return std::nullopt;
  }
}

std::optional<op> registry::get_operation(std::string const& prefix) {
  if (auto const it = operations_.upper_bound(prefix);
      it != begin(operations_) &&
      boost::algorithm::starts_with(prefix, std::next(it, -1)->first)) {
    return std::next(it, -1)->second;
  } else {
    return std::nullopt;
  }
}

std::optional<std::string_view> registry::get_operation_name(
    std::string const& prefix) {
  if (auto const it = operations_.upper_bound(prefix);
      it != begin(operations_) &&
      boost::algorithm::starts_with(prefix, std::next(it, -1)->first)) {
    return std::string_view{std::next(it, -1)->first};
  } else {
    std::lock_guard const g{remote_op_mutex_};
    if (auto const it = remote_operations_.upper_bound(prefix);
        it != begin(remote_operations_) &&
        boost::algorithm::starts_with(prefix, std::next(it, -1)->first)) {
      return std::string_view{std::next(it, -1)->first};
    } else {
      return std::nullopt;
    }
  }
}

void registry::reset() {
  operations_.clear();
  topic_subscriptions_.clear();
  remote_operations_.clear();
}

}  // namespace motis::module
