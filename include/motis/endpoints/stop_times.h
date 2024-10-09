#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct stop_times {
  api::stoptimes_response operator()(boost::urls::url_view const&) const;

  nigiri::timetable const& tt_;
  tag_lookup const& tags_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep