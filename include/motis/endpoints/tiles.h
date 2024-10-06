#pragma once

#include "net/web_server/query_router.h"

#include "motis/fwd.h"

namespace motis::ep {

struct tiles {
  net::reply operator()(net::route_request const&, bool) const;

  tiles_data& tiles_data_;
};

}  // namespace motis::ep