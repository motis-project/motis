#pragma once

#include "boost/json/value.hpp"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "osr/lookup.h"
#include "osr/ways.h"

namespace motis::ep {

struct nearest {
  api::NearestResponse operator()(boost::urls::url_view const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  config const& c_;
};

}  // namespace motis::ep
