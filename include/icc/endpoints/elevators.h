#pragma once

#include "boost/json/value.hpp"

#include "nigiri/types.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "icc/elevators/elevators.h"

namespace icc::ep {

struct elevators {
  boost::json::value operator()(boost::json::value const&) const;

  elevators_ptr_t const& e_;
  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace icc::ep