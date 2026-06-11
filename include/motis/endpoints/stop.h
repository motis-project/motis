#pragma once

#include "boost/url/url_view.hpp"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis::ep {

struct stop {
  api::stopInfo_response operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  adr::typeahead const* t_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  tag_lookup const& tags_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep
