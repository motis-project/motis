#pragma once

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"

namespace motis::ep {

struct one_to_all {
  api::Reachable operator()(boost::urls::url_view const&) const;

  osr::ways const* w_;
  osr::platforms const* pl_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
  tag_lookup const& tags_;
  platform_matches_t const* matches_;
};

}  // namespace motis::ep
