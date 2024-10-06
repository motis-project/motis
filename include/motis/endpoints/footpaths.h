#pragma once

#include "boost/url/url_view.hpp"

#include "nigiri/types.h"

#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis::ep {

struct footpaths {
  api::footpaths_response operator()(boost::urls::url_view const&) const;

  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  platform_matches_t const& matches_;
  std::shared_ptr<rt> rt_;
};

}  // namespace motis::ep