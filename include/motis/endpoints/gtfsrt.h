#pragma once

#include "net/web_server/query_router.h"

#include "motis/fwd.h"

namespace motis::ep {

struct gtfsrt {
  net::reply operator()(net::route_request const&, bool) const;

  config const& config_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  feed_labels const& labels_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep