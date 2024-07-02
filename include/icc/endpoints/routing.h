#pragma once

#include "boost/json/value.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "icc-api/icc-api.h"

namespace icc::ep {

struct routing {
  api::plan_response operator()(boost::urls::url_view const& url) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  nigiri::timetable const& tt_;
};

}  // namespace icc::ep