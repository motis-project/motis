#pragma once

#include "boost/json/value.hpp"

#include "nigiri/timetable.h"

#include "icc/fwd.h"
#include "icc/point_rtree.h"

namespace icc::ep {

struct matches {
  boost::json::value operator()(boost::json::value const&) const;

  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
};

}  // namespace icc::ep