#pragma once

#include <map>

#include "flatbuffers/flatbuffers.h"

#include "motis/hash_map.h"

#include "motis/schedule/bitfield.h"

namespace motis::loader::hrd {

struct bitfield_builder {
  static constexpr int no_bitfield_num = -1;

  explicit bitfield_builder(std::map<int, bitfield>);

  flatbuffers64::Offset<flatbuffers64::String> get_or_create_bitfield(
      int bitfield_num, flatbuffers64::FlatBufferBuilder&);

  flatbuffers64::Offset<flatbuffers64::String> get_or_create_bitfield(
      bitfield const&, flatbuffers64::FlatBufferBuilder&,
      int = no_bitfield_num);

  std::map<int, bitfield> hrd_bitfields_;
  mcd::hash_map<bitfield, flatbuffers64::Offset<flatbuffers64::String>,
                std::hash<bitfield>, std::equal_to<>>
      fbs_bitfields_;
  std::map<int, flatbuffers64::Offset<flatbuffers64::String>> fbs_bf_lookup_;
};

}  // namespace motis::loader::hrd
