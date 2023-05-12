#include "motis/rt/message_history.h"

namespace motis::rt {

void message_history::add(trip_idx_t const trip_idx,
                          std::string_view const msg_buffer) {
  if (enabled_) {
    messages_[trip_idx].emplace_back(msg_buffer);
  }
}

}  // namespace motis::rt
