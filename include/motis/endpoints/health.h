#pragma once

#include <string_view>

#include "boost/url/url_view.hpp"

#include "motis/config.h"
#include "motis/metrics_registry.h"

namespace motis::ep {

struct health {
  std::string_view operator()(boost::urls::url_view const&) const;

  metrics_registry const* metrics_;
  config const& config_;
};

}  // namespace motis::ep