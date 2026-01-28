#pragma once

#include <optional>

#include "net/web_server/query_router.h"

#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"

namespace motis::ep {

struct ojp {
  net::reply operator()(net::route_request const&, bool) const;

  std::optional<routing> r_;
  std::optional<geocode> geocoding_;
  std::optional<stops> s_;
  std::optional<stop_times> st_;
};

}  // namespace motis::ep