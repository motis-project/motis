#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "icc/elevators/elevators.h"
#include "icc/fwd.h"

namespace icc::ep {

struct elevators {
  boost::json::value operator()(boost::json::value const&) const;

  std::shared_ptr<rt> const& rt_;
  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace icc::ep