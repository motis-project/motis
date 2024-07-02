#pragma once

#include "boost/json/value.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/ways.h"

namespace icc::ep {

struct routing {
  boost::json::value operator()(boost::json::value const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  nigiri::timetable const& tt_;
};

}  // namespace icc::ep