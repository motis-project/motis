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

// One one-to-many street search of a request, kept so that the offset legs of
// journeys and their alternatives can be reconstructed from it instead of
// being routed again.
struct one_to_many_search {
  std::unique_ptr<osr::one_to_many_state> state_;
  hash_map<nigiri::location_idx_t, std::size_t> dest_idx_;  // stop -> dest
  // Flex only: the additional nodes the search ran with. The routing rebuilds
  // its own for the next flex group, so the legs reconstructed from this
  // search are rendered with these.
  flex::flex_additional_nodes flex_additional_nodes_;
};

// The retained searches of one place. Several transport modes can share a
// search: a flex routing group runs one search for every transport of a stop
// sequence that boards at the same stop.
struct one_to_many_side {
  std::vector<one_to_many_search> searches_;
  hash_map<transport_mode_t, std::size_t> by_mode_;
};

struct one_to_many_searches {
  // Only `kStart` and `kEnd` address a side; no offsets are computed from a
  // via station.
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

// Reconstruct destination `dest_idx_` of `state_` instead of routing.
struct precomputed_route {
  osr::one_to_many_state* state_{nullptr};
  std::size_t dest_idx_{0U};
  // Borrowed from the `one_to_many_search` that owns them.
  flex::flex_additional_nodes const* flex_additional_nodes_{nullptr};
};

// A request's retained searches as seen from one journey. `flipped_` maps the
// journey's sides onto the sides the offsets were computed for: alternatives
// of an arriveBy query are rendered from a direction-flipped query, where the
// special start station refers to the journey's destination place.
struct one_to_many_view {
  // Returns an empty route unless the leg touches the journey's start or end
  // place and that side has a search covering `target`.
  precomputed_route find(nigiri::location_idx_t leg_from,
                         nigiri::location_idx_t leg_to,
                         transport_mode_t,
                         nigiri::location_idx_t target) const;

  one_to_many_searches const* searches_{nullptr};
  bool flipped_{false};
};

}  // namespace motis
