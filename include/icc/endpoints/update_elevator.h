#pragma once

#include "boost/json/value.hpp"

#include "osr/types.h"
#include "osr/ways.h"

#include "icc/elevators/elevators.h"
#include "icc/types.h"

namespace icc::ep {

struct update_elevator {
  boost::json::value operator()(boost::json::value const&) const;

  shared_elevators& e_;
  osr::ways const& w_;
  hash_set<osr::node_idx_t> const& elevator_nodes_;
};

}  // namespace icc::ep