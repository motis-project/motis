#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct suspects {
  api::plan_response operator()(boost::urls::url_view const&) const;

  ::motis::suspects const& suspects_;
  nigiri::timetable const& tt_;
};

}  // namespace motis::ep