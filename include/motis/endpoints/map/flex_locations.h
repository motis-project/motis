#pragma once

#include "boost/json/value.hpp"
#include "boost/url/url_view.hpp"

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis::ep {

struct flex_locations {
  boost::json::value operator()(boost::urls::url_view const&) const;

  tag_lookup const& tags_;
  nigiri::timetable const& tt_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
};

}  // namespace motis::ep