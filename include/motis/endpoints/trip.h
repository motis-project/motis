#pragma once

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis::ep {

struct trip {
  api::Itinerary operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  nigiri::timetable const& tt_;
  nigiri::shapes_storage const* shapes_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  tag_lookup const& tags_;
  point_rtree<nigiri::location_idx_t> const& loc_tree_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep