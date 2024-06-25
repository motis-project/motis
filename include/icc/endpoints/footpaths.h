#pragma once

#include "boost/json/value.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "icc/elevators/elevators.h"
#include "icc/point_rtree.h"
#include "icc/types.h"

namespace icc::ep {

struct footpaths {
  boost::json::value operator()(boost::json::value const&) const;

  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches_;
  shared_elevators const& e_;
};

}  // namespace icc::ep