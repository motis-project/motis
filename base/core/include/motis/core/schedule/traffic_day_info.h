#pragma once

#include <algorithm>
#include <tuple>

#include "motis/pair.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/bitfield.h"

namespace motis {

template <typename T>
struct traffic_day_info {
  traffic_day_info() = default;

  explicit traffic_day_info(
      T default_entry, mcd::vector<mcd::pair<bitfield_idx_t, T>> entries = {})
      : default_{std::move(default_entry)},
        entries_{mcd::to_vec(entries, [](auto&& pair) {
          return mcd::pair{bitfield_idx_or_ptr{.bitfield_idx_ = pair.first},
                           pair.second};
        })} {}

  auto cista_members() { return std::tie(default_, entries_); }

  T get_info(day_idx_t const day) const {
    auto const it =
        std::find_if(begin(entries_), end(entries_), [&](auto&& entry) {
          auto const& [traffic_days, el] = entry;
          return traffic_days.traffic_days_->test(day);
        });
    return it == end(entries_) ? default_ : it->second;
  }

  T default_{};
  mcd::vector<mcd::pair<bitfield_idx_or_ptr, T>> entries_;
};

}  // namespace motis