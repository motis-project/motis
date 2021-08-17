#pragma once

#include <map>
#include <mutex>
#include <optional>

#include "ctx/access_t.h"

#include "motis/module/client.h"
#include "motis/module/message.h"
#include "motis/module/receiver.h"

namespace motis::module {

using void_op_fn_t = std::function<void()>;
using op_fn_t = std::function<msg_ptr(msg_ptr const&)>;
using remote_op_fn_t = std::function<void(msg_ptr, callback)>;

struct op {
  op(std::function<msg_ptr(msg_ptr const&)> fn, ctx::access_t access)
      : fn_{std::move(fn)}, access_{access} {}
  op_fn_t fn_;
  ctx::access_t access_;
};

struct registry {
  void register_op(std::string const& name, op_fn_t,
                   ctx::access_t access = ctx::access_t::READ);

  void register_client_handler(std::string&& target,
                               std::function<void(client_hdl)>&&);

  void subscribe(std::string const& topic, op_fn_t,
                 ctx::access_t access = ctx::access_t::READ);

  void subscribe(std::string const& topic, void_op_fn_t,
                 ctx::access_t access = ctx::access_t::READ);

  std::vector<std::string> register_remote_ops(
      std::vector<std::string> const& names, remote_op_fn_t const& fn);

  void unregister_remote_op(std::vector<std::string> const& names);

  std::optional<remote_op_fn_t> get_remote_op(std::string const& prefix);

  std::optional<op> get_operation(std::string const& prefix);

  void reset();

  std::map<std::string, op> operations_;
  std::map<std::string, std::vector<op>> topic_subscriptions_;
  std::map<std::string, std::function<void(client_hdl)>> client_handlers_;

  std::mutex mutable remote_op_mutex_;
  std::map<std::string, remote_op_fn_t> remote_operations_;
};

}  // namespace motis::module
