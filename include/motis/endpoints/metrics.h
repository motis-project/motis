#pragma once

#include <string_view>

#include "net/web_server/query_router.h"

#include "prometheus/registry.h"

namespace motis::ep {

struct metrics {
  net::reply operator()(net::route_request const&, bool) const;

  prometheus::Registry const& metrics_;
};

}  // namespace motis::ep