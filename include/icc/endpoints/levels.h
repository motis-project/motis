#pragma once

#include "boost/json/value.hpp"

#include "icc/fwd.h"

namespace icc::ep {

struct levels {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace icc::ep