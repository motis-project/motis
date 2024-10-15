#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis::ep {

struct matches {
  boost::json::value operator()(boost::json::value const&) const;

  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  tag_lookup const& tags_;
  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
};

}  // namespace motis::ep