#pragma once

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"

namespace motis::ep {

struct one_to_all {
  api::Reachable operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
  tag_lookup const& tags_;
  flex::flex_areas const* fa_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  platform_matches_t const* matches_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  way_matches_storage const* way_matches_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep
