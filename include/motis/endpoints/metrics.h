#pragma once

#include <string_view>

#include "net/web_server/query_router.h"

#include "motis/fwd.h"
#include "motis/metrics_registry.h"

namespace motis::ep {

struct metrics {
  net::reply operator()(net::route_request const&, bool) const;

  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  std::shared_ptr<rt> const& rt_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep