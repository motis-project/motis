#pragma once

#include "boost/json/value.hpp"

#include "icc/data.h"

namespace icc::ep {

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

}  // namespace icc::ep