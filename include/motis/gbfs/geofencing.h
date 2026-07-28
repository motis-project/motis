#pragma once

#include <string>
#include <vector>

#include "tg.h"

#include "geo/latlng.h"

#include "motis/gbfs/data.h"

namespace motis::gbfs {

bool applies(std::vector<vehicle_type_idx_t> const& rule_vehicle_type_idxs,
             std::vector<vehicle_type_idx_t> const& segment_vehicle_type_idxs);
bool multipoly_contains_point(tg_geom const* geom, geo::latlng const& pos);

template <typename Fn>
void multipoly_split_bboxes(tg_geom const* geom, Fn&& fn) {
  auto const n_polys = tg_geom_num_polys(geom);
  for (auto i = 0; i < n_polys; ++i) {
    auto const* poly = tg_geom_poly_at(geom, i);
    auto rect = tg_poly_rect(poly);
    fn(geo::box{geo::latlng{rect.min.y, rect.min.x},
                geo::latlng{rect.max.y, rect.max.x}});
  }
}

geofencing_restrictions get_default_restrictions(gbfs_provider const&,
                                                 provider_products const&,
                                                 std::vector<rule> const&);
bool vehicle_is_rentable(gbfs_provider const&,
                         provider_products const&,
                         vehicle_status const&);

}  // namespace motis::gbfs
