#pragma once

#include <utility>

#include "boost/beast/http/status.hpp"
#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/metrics_registry.h"

namespace motis::ep {

struct health {
  using response_t = std::pair<boost::beast::http::status, api::HealthResponse>;
  response_t operator()(boost::urls::url_view const&) const;

  config const& config_;
  metrics_registry const* metrics_;
};

}  // namespace motis::ep