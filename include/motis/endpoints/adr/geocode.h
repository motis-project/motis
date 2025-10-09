#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis::ep {

struct geocode {
  api::geocode_response operator()(boost::urls::url_view const& url) const;

  osr::ways const* w_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  adr::typeahead const& t_;
  adr::formatter const& f_;
  adr::cache& cache_;
  adr_ext const* ae_;
};

}  // namespace motis::ep