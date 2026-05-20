#pragma once

#include <cinttypes>
#include <memory>
#include <optional>
#include <vector>

#include "geo/latlng.h"

#include "cista/reflection/comparable.h"
#include "cista/strong.h"

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

#include "motis/elevators/get_state_changes.h"

namespace nigiri {
struct rt_timetable;
}

namespace motis {

template <typename K, typename V>
using vector_map = nigiri::vector_map<K, V>;

template <typename T>
using hash_set = nigiri::hash_set<T>;

template <typename K,
          typename V,
          typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = nigiri::hash_map<K, V, Hash, Equality>;

template <typename T>
using basic_string = std::basic_string<T, cista::char_traits<T>>;

using elevator_idx_t = cista::strong<std::uint32_t, struct elevator_idx_>;

using gbfs_provider_idx_t =
    cista::strong<std::uint16_t, struct gbfs_provider_idx_>;

struct elevator {
  friend bool operator==(elevator const&, elevator const&) = default;

  std::vector<state_change<nigiri::unixtime_t>> const& get_state_changes()
      const {
    return state_changes_;
  }

  std::int64_t id_;
  std::optional<std::string> id_str_;
  geo::latlng pos_;
  bool status_;
  std::string desc_;
  std::vector<nigiri::interval<nigiri::unixtime_t>> out_of_service_;
  std::vector<state_change<nigiri::unixtime_t>> state_changes_{
      intervals_to_state_changes(out_of_service_, status_)};
};

using rtt_ptr_t = std::shared_ptr<nigiri::rt_timetable>;

}  // namespace motis
