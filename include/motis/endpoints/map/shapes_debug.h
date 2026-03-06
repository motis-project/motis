#pragma once

#include "net/web_server/query_router.h"

#include "motis/config.h"
#include "motis/fwd.h"

namespace motis::ep {

struct shapes_debug {
  net::reply operator()(net::route_request const&, bool) const;

  config const& c_;
  osr::ways const* w_;
  osr::lookup const* l_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
};

}  // namespace motis::ep
