#include "motis/chat/chat.h"

#include "motis/core/common/logging.h"

namespace motis::chat {

void chat::init(motis::module::registry& reg) {
  reg.client_handlers_.emplace(
      "/chat", [&](motis::module::client_hdl const& c) {
        if (auto const lock = c.lock(); lock) {
          lock->set_on_close_cb([this, c]() {
            auto const lock = c.lock();
            auto const it =
                std::find_if(begin(clients_), end(clients_),
                             [&lock](auto&& o) { return o.lock() == lock; });
            if (it != end(clients_)) {
              LOG(motis::logging::info) << " removing client";
              clients_.erase(it);
            }
          });
          lock->set_on_msg_cb([this, c](motis::module::msg_ptr const& msg) {
            auto msgs_sent = 0U;

            auto const lock = c.lock();
            for (auto const& o : clients_) {
              if (auto const o_lock = o.lock(); o_lock != lock) {
                o_lock->send(motis::module::make_success_msg(
                    msg->get()->destination()->target()->str()));
                ++msgs_sent;
              }
            }

            LOG(motis::logging::info)
                << " broadcasted to " << msgs_sent << " / " << clients_.size();
          });
          clients_.emplace_back(c);
          LOG(motis::logging::error) << " client added";
        } else {
          LOG(motis::logging::error) << " client obsolete";
        }
      });
}

}  // namespace motis::chat
