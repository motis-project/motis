#pragma once

#include <memory>
#include <queue>
#include <string_view>

#include "ctx/access_scheduler.h"

#include "motis/module/ctx_data.h"
#include "motis/module/future.h"
#include "motis/module/message.h"
#include "motis/module/module.h"
#include "motis/module/receiver.h"
#include "motis/module/registry.h"
#include "motis/module/shared_data.h"

namespace motis::module {

/* Concurrency model:
 *   - multiple reads or one single write at a time
 *   - parent op needs to have equal or more permissions as child
 *   - parent op may not quit until all child ops that require
 *     access permissions have finished
 *   - (internal) -> child ops do not require own access permissions
 *   - note: write ops could spawn multiple parallel write child
 *           ops which leads to (unchecked) data-races!
 */
struct dispatcher : public receiver, public ctx::access_scheduler<ctx_data> {
  explicit dispatcher(registry&, std::vector<std::unique_ptr<module>>&&);

  template <typename Module>
  Module& get_module(std::string_view module_name) {
    auto it = std::find_if(
        modules_.begin(), modules_.end(),
        [&module_name](auto const& m) { return m->name() == module_name; });
    if (it == modules_.end()) {
      throw std::runtime_error("module not found");
    }
    return *reinterpret_cast<Module*>(it->get());
  }

  std::vector<future> publish(msg_ptr const& msg, ctx_data const& data,
                              ctx::op_id id);

  future req(msg_ptr const& msg, ctx_data const& data, ctx::op_id const& id);

  void on_msg(msg_ptr const& msg, callback const& cb) override;
  void on_connect(std::string const& target, client_hdl const&) override;

  void dispatch(msg_ptr const& msg, callback const& cb, ctx::op_id id,
                ctx::op_type_t op_type, ctx_data const* data = nullptr);

  motis::module::msg_ptr api_desc(int id) const;

  ctx::access_t access_of(std::string const& target);
  ctx::access_t access_of(msg_ptr const& msg);

  void handle_no_target(msg_ptr const& msg, callback const& cb);
  void retry_no_target_msgs();

  registry& registry_;
  bool queue_no_target_msgs_{false};
  std::queue<std::pair<msg_ptr, callback>> no_target_msg_queue_;
  std::vector<std::unique_ptr<module>> modules_;
  shared_data shared_data_;

  // If this is set to a value != nullptr, it indicates direct mode is on.
  // This implies that in direct mode there can only be one global dispatcher.
  // Direct mode means that
  //   - no code will be executed in ctx::operations.
  //   - no calls to ctx::current_op<Data>() will be made
  //   - everything runs sequentially (no interleaving for motis_call/publish)
  static dispatcher* direct_mode_dispatcher_;
};

}  // namespace motis::module
