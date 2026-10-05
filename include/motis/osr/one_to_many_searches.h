#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <vector>

#include "osr/routing/route.h"

#include "nigiri/special_stations.h"

#include "motis/flex/flex_routing_data.h"
#include "motis/transport_mode.h"
#include "motis/types.h"

namespace motis {

struct one_to_many_search {
  std::unique_ptr<osr::one_to_many_state> state_;
  hash_map<nigiri::location_idx_t, std::size_t> dest_idx_;
  flex::flex_additional_nodes flex_additional_nodes_;
};

struct one_to_many_side {
  std::vector<one_to_many_search> searches_;
  hash_map<transport_mode_t, std::size_t> by_mode_;
};

struct one_to_many_searches {
  static std::size_t idx(nigiri::special_station const s) {
    assert(s == nigiri::special_station::kStart ||
           s == nigiri::special_station::kEnd);
    return static_cast<std::size_t>(s);
  }

  one_to_many_side const& operator[](nigiri::special_station const s) const {
    return sides_[idx(s)];
  }
  one_to_many_side& operator[](nigiri::special_station const s) {
    return sides_[idx(s)];
  }

  std::array<one_to_many_side, 2> sides_;
};

struct precomputed_route {
  osr::one_to_many_state* state_{nullptr};
  std::size_t dest_idx_{0U};
  flex::flex_additional_nodes const* flex_additional_nodes_{nullptr};
};

struct one_to_many_view {
  precomputed_route find(nigiri::location_idx_t leg_from,
                         nigiri::location_idx_t leg_to,
                         transport_mode_t,
                         nigiri::location_idx_t target) const;

  one_to_many_searches const* searches_{nullptr};
  bool arrive_by_{false};
};

}  // namespace motis
