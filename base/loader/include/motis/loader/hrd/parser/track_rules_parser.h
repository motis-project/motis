#pragma once

#include <cinttypes>
#include <map>
#include <tuple>

#include "flatbuffers/flatbuffers.h"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

constexpr int TIME_NOT_SET = -1;
struct track_rule {
  flatbuffers64::Offset<flatbuffers64::String> track_name_;
  int bitfield_num_{0};
  int time_{0};
};

using track_rule_key = std::tuple<int, int, uint64_t>;
using track_rules = std::map<track_rule_key, std::vector<track_rule>>;

track_rules parse_track_rules(loaded_file const&,
                              flatbuffers64::FlatBufferBuilder& b,
                              config const&);

}  // namespace motis::loader::hrd
