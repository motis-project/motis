#pragma once

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct levels {
  api::levels_response operator()(boost::urls::url_view const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
};

}  // namespace motis::ep