#pragma once

#include "boost/json/value.hpp"

#include "icc/data.h"
#include "icc/elevators/elevators.h"

namespace icc::ep {

struct osr_routing {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace icc::ep