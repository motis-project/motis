#pragma once

#include "boost/json/value.hpp"

#include "motis/fwd.h"

namespace motis::ep {

struct graph {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace motis::ep