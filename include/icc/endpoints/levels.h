#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "icc/point_rtree.h"
#include "icc/types.h"

namespace icc::ep {

struct levels {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace icc::ep