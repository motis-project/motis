#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct route_search {
  api::routeSearch_response operator()(boost::urls::url_view const&) const;

  config const& config_;
  nigiri::timetable const& tt_;
  nigiri::shapes_storage const* shapes_;
};

}  // namespace motis::ep
