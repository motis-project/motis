#pragma once

#include "net/web_server/query_router.h"

#include "motis/config.h"
#include "motis/metrics_registry.h"

namespace motis::ep {

struct health {
  net::reply operator()(net::route_request const&, bool) const;

  config const& config_;
  metrics_registry const* metrics_;
};

}  // namespace motis::ep