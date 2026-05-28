#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"

namespace nigiri {
using location_idx_t = cista::strong<uint32_t, struct _location_idx>;
}

namespace motis::ep {

struct geocode {
  api::geocode_response operator()(boost::urls::url_view const& url) const;

  config const& config_;
  osr::ways const* w_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  adr::typeahead const& t_;
  adr::formatter const& f_;
  adr::cache& cache_;
  adr_ext const* ae_;
  point_rtree<nigiri::location_idx_t> const* location_rtree_{nullptr};
};

}  // namespace motis::ep