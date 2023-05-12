#pragma once

#include <string_view>
#include <vector>

#include "motis/hash_map.h"

#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/schedule/trip.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::rt {

struct message_history {
  explicit message_history(bool enabled) : enabled_{enabled} {}

  void add(trip_idx_t trip_idx, std::string_view msg_buffer);

  bool enabled_{};
  mcd::hash_map<trip_idx_t,
                std::vector<typed_flatbuffer<motis::ris::RISMessage>>>
      messages_;
};

}  // namespace motis::rt
