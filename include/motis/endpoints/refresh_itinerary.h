#pragma once

#include <memory>

#include "boost/json.hpp"
#include "boost/url/url_view.hpp"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis::ep {

struct refresh_itinerary {
  api::Itinerary operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const& tt_;
  nigiri::routing::tb::tb_data const* tbd_;
  tag_lookup const& tags_;
  point_rtree<nigiri::location_idx_t> const& loc_tree_;
  flex::flex_areas const* fa_;
  platform_matches_t const* matches_;
  way_matches_storage const* way_matches_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  adr::typeahead const* t_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  odm::bounds const* odm_bounds_;
  odm::ride_sharing_bounds const* ride_sharing_bounds_;
  metrics_registry* metrics_;
};

struct refresh_itinerary_post {
  api::Itinerary operator()(boost::urls::url_view const&,
                            api::RefreshItineraryPostBody const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const& tt_;
  nigiri::routing::tb::tb_data const* tbd_;
  tag_lookup const& tags_;
  point_rtree<nigiri::location_idx_t> const& loc_tree_;
  flex::flex_areas const* fa_;
  platform_matches_t const* matches_;
  way_matches_storage const* way_matches_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  adr::typeahead const* t_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  odm::bounds const* odm_bounds_;
  odm::ride_sharing_bounds const* ride_sharing_bounds_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep
