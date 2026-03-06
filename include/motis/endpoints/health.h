#pragma once

#include "net/web_server/query_router.h"
#include "boost/url/url_view.hpp"

#include "motis/metrics_registry.h"

namespace motis::ep {

struct health {
  // motis up and running
  // motis consumed rt data (if enabled)

  net::reply operator()(boost::urls::url_view const&) const;

  metrics_registry const* metrics_;
};

}  // namespace motis::ep