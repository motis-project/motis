#pragma once

#include "boost/json/value.hpp"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "nigiri/timetable.h"

#include "icc/elevators/elevators.h"
#include "icc/types.h"

namespace icc::ep {

struct update_elevator {
  boost::json::value operator()(boost::json::value const&) const;

  nigiri::timetable const& tt_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  hash_set<osr::node_idx_t> const& elevator_nodes_;
  vector_map<nigiri::location_idx_t, osr::platform_idx_t> matches_;
  elevators_ptr_t& e_;
  rtt_ptr_t& rtt_;
};

}  // namespace icc::ep