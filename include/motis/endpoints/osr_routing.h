#pragma once

#include "boost/json/value.hpp"

#include "motis/elevators/elevators.h"
#include "motis/fwd.h"

namespace motis::ep {

struct osr_routing {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep