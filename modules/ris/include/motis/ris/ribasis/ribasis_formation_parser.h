#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "flatbuffers/flatbuffers.h"

#include "rapidjson/document.h"

#include "motis/hash_map.h"
#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/ribasis/common.h"

namespace motis::ris::ribasis::formation {

struct context {
  explicit context(ris_msg_context& ris_ctx) : ris_{ris_ctx} {}

  ris_msg_context& ris_;

  mcd::hash_map<std::string, flatbuffers::Offset<StationInfo>> stations_;
  flatbuffers::Offset<HalfTripId> half_trip_id_{};

  std::uint32_t fahrtnummer_{};
  std::string_view gattung_;
};

void parse_ribasis_formation(ris_msg_context& ris_ctx,
                             rapidjson::Value const& data);

}  // namespace motis::ris::ribasis::formation
