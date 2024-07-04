#pragma once

#include "boost/json/value.hpp"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "icc/elevators/elevators.h"

namespace icc::ep {

struct osr_routing {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  elevators_ptr_t const& e_;
};

}  // namespace icc::ep