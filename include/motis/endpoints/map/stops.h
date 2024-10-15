#pragma once

#include "boost/url/url_view.hpp"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis::ep {

struct stops {
  api::stops_response operator()(boost::urls::url_view const&) const;

  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  tag_lookup const& tags_;
  nigiri::timetable const& tt_;
};

}  // namespace motis::ep