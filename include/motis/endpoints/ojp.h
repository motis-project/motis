#pragma once

#include <optional>

#include "net/web_server/query_router.h"

#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/trip.h"

namespace motis::ep {

struct ojp {
  net::reply operator()(net::route_request const&, bool) const;

  std::optional<routing> routing_ep_;
  std::optional<geocode> geocoding_ep_;
  std::optional<stops> stops_ep_;
  std::optional<stop_times> stop_times_ep_;
  std::optional<trip> trip_ep_;
};

}  // namespace motis::ep