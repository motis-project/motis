#pragma once

#include <cinttypes>
#include <map>
#include <vector>

#include "motis/schedule-format/Direction_generated.h"

#include "motis/loader/hrd/builder/station_builder.h"

namespace motis::loader::hrd {

struct direction_builder {
  explicit direction_builder(std::map<uint64_t, std::string>);

  flatbuffers64::Offset<Direction> get_or_create_direction(
      std::vector<std::pair<uint64_t, int>> const&, station_builder&,
      flatbuffers64::FlatBufferBuilder&);

  std::map<uint64_t, std::string> hrd_directions_;
  std::map<uint64_t, flatbuffers64::Offset<Direction>> fbs_directions_;
};

}  // namespace motis::loader::hrd
