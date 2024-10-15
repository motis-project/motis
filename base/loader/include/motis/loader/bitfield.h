#pragma once

#include <cassert>
#include <algorithm>
#include <bitset>

#include "utl/parser/cstr.h"

namespace motis::loader {

constexpr int BIT_COUNT = 2048;
using bitfield = std::bitset<BIT_COUNT>;

template <std::size_t BitSetSize>
struct bitset_comparator {
  bool operator()(std::bitset<BitSetSize> const& lhs,
                  std::bitset<BitSetSize> const& rhs) const {
    for (std::size_t i = 0; i < BitSetSize; ++i) {
      int lhs_bit = lhs.test(i) ? 1 : 0;
      int rhs_bit = rhs.test(i) ? 1 : 0;
      if (lhs_bit != rhs_bit) {
        return lhs_bit < rhs_bit;
      }
    }
    return false;
  }
};

template <std::size_t BitSetSize>
std::bitset<BitSetSize> create_uniform_bitfield(char val) {
  assert(val == '1' || val == '0');

  std::string all_days_bit_str;
  all_days_bit_str.resize(BitSetSize);
  std::fill(begin(all_days_bit_str), end(all_days_bit_str), val);

  return std::bitset<BitSetSize>(all_days_bit_str);
}

template <std::size_t BitCount>
inline std::string serialize_bitset(std::bitset<BitCount> const& bitset) {
  return bitset.to_string();
}

template <std::size_t BitCount>
inline std::bitset<BitCount> deserialize_bitset(utl::cstr str) {
  return std::bitset<BitCount>(std::string(str.str, str.len));
}

}  // namespace motis::loader
