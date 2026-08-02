#pragma once

#include <optional>
#include <string>

#include "net/web_server/query_router.h"

#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/routing.h"

namespace motis::ep {

struct mcp {
  net::reply operator()(net::route_request const&, bool) const;

  std::optional<routing> routing_ep_;
  std::optional<geocode> geocoding_ep_;
  std::string motis_version_;
};

}  // namespace motis::ep
