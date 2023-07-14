#include "motis/nigiri/station_lookup.h"

#include "geo/point_rtree.h"

#include "nigiri/timetable.h"

#include "motis/nigiri/location.h"
#include "motis/nigiri/tag_lookup.h"

namespace n = nigiri;

namespace motis::nigiri {

nigiri_station_lookup::nigiri_station_lookup(
    motis::nigiri::tag_lookup const& tags, ::nigiri::timetable const& tt)
    : station_lookup{geo::make_point_rtree(tt.locations_.coordinates_)},
      tt_{tt},
      tags_{tags} {}

nigiri_station_lookup::~nigiri_station_lookup() noexcept = default;

lookup_station nigiri_station_lookup::get(std::size_t const idx) const {
  auto const l = n::location_idx_t{idx};
  auto const p = tt_.locations_.parents_[l];
  auto s = lookup_station{};
  s.tag_ = tags_.get_tag(tt_.locations_.src_[l]),
  s.id_ = tt_.locations_.ids_[l].view(),
  s.name_ = (p == n::location_idx_t::invalid() ? tt_.locations_.names_[l]
                                               : tt_.locations_.names_[p])
                .view(),
  s.pos_ = tt_.locations_.coordinates_[l];
  return s;
}

lookup_station nigiri_station_lookup::get(std::string_view id) const {
  try {
    return get(to_idx(get_location_idx(tags_, tt_, id)));
  } catch (...) {
    return lookup_station{};
  }
}

}  // namespace motis::nigiri