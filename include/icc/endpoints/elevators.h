#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "icc/point_rtree.h"
#include "icc/types.h"

namespace icc::ep {

struct elevators {
  boost::json::value operator()(boost::json::value const&) const;

  point_rtree<elevator_idx_t> const& elevators_rtree_;
  nigiri::vector_map<elevator_idx_t, elevator> elevators_;
  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace icc::ep