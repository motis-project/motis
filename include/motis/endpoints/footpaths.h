#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "osr/types.h"

#include "motis/data.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis::ep {

struct footpaths {
  boost::json::value operator()(boost::json::value const&) const;

  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches_;
  std::shared_ptr<rt> rt_;
};

}  // namespace motis::ep