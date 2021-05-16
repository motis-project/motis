#pragma once

#include <cassert>
#include <algorithm>
#include <bitset>

#include "utl/parser/cstr.h"

namespace motis::loader {

constexpr unsigned BIT_COUNT = 512;
using bitfield = std::bitset<BIT_COUNT>;

template <std::size_t BitSetSize>
bool operator<(std::bitset<BitSetSize> const& x,
               std::bitset<BitSetSize> const& y) {
  // i > 0
  for (auto i = size_t{BitSetSize - 1}; i != 0; --i) {
    if (x[i] ^ y[i]) {
      return y[i];
    }
  }
  // i = 0
  if (x[0] ^ y[0]) {
    return y[0];
  }
  return false;
}

template <std::size_t BitSetSize>
struct bitset_comparator {
  bool operator()(std::bitset<BitSetSize> const& lhs,
                  std::bitset<BitSetSize> const& rhs) const {
    return lhs < rhs;
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
  return std::bitset<BitCount>(str.str, str.len);
}

}  // namespace motis::loader
