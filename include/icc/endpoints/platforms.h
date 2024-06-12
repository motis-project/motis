#pragma once

#include "boost/json/value.hpp"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

namespace icc::ep {

struct platforms {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
};

}  // namespace icc::ep