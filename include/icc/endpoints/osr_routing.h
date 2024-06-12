#pragma once

#include "boost/json/value.hpp"

#include "osr/lookup.h"
#include "osr/ways.h"

namespace icc::ep {

struct osr_routing {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  std::shared_ptr<osr::bitvec<osr::node_idx_t>> blocked_;
};

}  // namespace icc::ep