#pragma once

#include <cassert>
#include <algorithm>

#include "cista/containers/bitset.h"

#include "utl/parser/cstr.h"

#include "motis/core/schedule/time.h"

namespace motis {

using bitfield = cista::bitset<MAX_DAYS>;

template <std::size_t BitSetSize>
bitfield create_uniform_bitfield(char val) {
  assert(val == '1' || val == '0');

  std::string all_days_bit_str;
  all_days_bit_str.resize(BitSetSize);
  std::fill(begin(all_days_bit_str), end(all_days_bit_str), val);

  return {all_days_bit_str};
}

inline std::string serialize_bitset(bitfield const& bitset) {
  return bitset.to_string();
}

inline bitfield deserialize_bitset(utl::cstr str) { return {str.view()}; }

}  // namespace motis
