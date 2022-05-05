#pragma once

#include <cassert>
#include <algorithm>
#include <limits>
#include <ostream>

#include "cista/containers/bitset.h"

#include "utl/parser/cstr.h"

#include "motis/core/schedule/time.h"
#include "motis/data.h"

namespace motis {

using bitfield = cista::bitset<MAX_DAYS>;

using bitfield_idx_t = size_t;

union bitfield_idx_or_ptr {
  bitfield_idx_or_ptr()
      : bitfield_idx_{std::numeric_limits<bitfield_idx_t>::max()} {}
  bitfield_idx_or_ptr(bitfield_idx_t const bf) : bitfield_idx_{bf} {}
  //  bitfield_idx_or_ptr(bitfield_idx_or_ptr const& o)
  //      : bitfield_idx_{o.bitfield_idx_} {}
  //  bitfield_idx_or_ptr(bitfield_idx_or_ptr&& o)
  //      : bitfield_idx_{o.bitfield_idx_} {}
  //  bitfield_idx_or_ptr& operator=(bitfield_idx_or_ptr const& o) {
  //    bitfield_idx_ = o.bitfield_idx_;
  //    return *this;
  //  }
  //  bitfield_idx_or_ptr& operator=(bitfield_idx_or_ptr&& o) {
  //    bitfield_idx_ = o.bitfield_idx_;
  //    return *this;
  //  }
  cista::hash_t hash() const { return bitfield_idx_; }

  bool operator==(std::nullptr_t) const { return traffic_days_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return traffic_days_ != nullptr; }
  bitfield const* operator->() const { return traffic_days_; }
  bitfield const& operator*() const { return *traffic_days_; }
  operator size_t() const { return bitfield_idx_; }  // NOLINT
  bitfield_idx_or_ptr& operator=(bitfield const* bf) {
    traffic_days_ = bf;
    return *this;
  }
  //  ptr<bitfield const> traffic_days_; // TODO(felix)
  bitfield const* traffic_days_;
  bitfield_idx_t bitfield_idx_;
};

inline void print(std::ostream& out, bitfield const& b) {
  out << "traffic_days={";
  auto first = true;
  for (auto i = day_idx_t{0}; i != MAX_DAYS; ++i) {
    if (b.test(i)) {
      if (!first) {
        out << ", ";
      } else {
        first = false;
      }
      out << i;
    }
  }
  out << "}";
}

inline bitfield shifted_bitfield(bitfield const& orig, day_idx_t const offset) {
  return offset > 0 ? orig << static_cast<std::size_t>(offset)
                    : orig >> static_cast<std::size_t>(-offset);
}

inline bitfield create_uniform_bitfield(char val) {
  assert(val == '1' || val == '0');

  std::string all_days_bit_str;
  all_days_bit_str.resize(MAX_DAYS);
  std::fill(begin(all_days_bit_str), end(all_days_bit_str), val);

  return bitfield{all_days_bit_str};
}

inline std::string serialize_bitset(bitfield const& bitset) {
  return bitset.to_string();
}

inline bitfield deserialize_bitset(utl::cstr str) { return {str.view()}; }

}  // namespace motis
