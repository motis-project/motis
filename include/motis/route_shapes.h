#pragma once

#include <optional>

#include "cista/containers/hash_map.h"

#include "geo/box.h"

#include "nigiri/types.h"

#include "osr/routing/profile.h"

#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

struct shape_cache_entry {
  bool valid() const {
    return shape_idx_ != nigiri::scoped_shape_idx_t::invalid();
  }

  nigiri::scoped_shape_idx_t shape_idx_{nigiri::scoped_shape_idx_t::invalid()};
  nigiri::vector<nigiri::shape_offset_t> offsets_;
  geo::box route_bbox_;
  nigiri::vector<geo::box> segment_bboxes_;
};

using shape_cache_key =
    nigiri::pair<osr::search_profile, nigiri::vector<geo::latlng>>;

using shape_cache_t = cista::raw::hash_map<shape_cache_key, shape_cache_entry>;

void route_shapes(osr::ways const&,
                  osr::lookup const&,
                  nigiri::timetable&,
                  nigiri::shapes_storage&,
                  config::timetable::route_shapes const&,
                  std::array<bool, nigiri::kNumClasses> const&,
                  shape_cache_t*);

}  // namespace motis
