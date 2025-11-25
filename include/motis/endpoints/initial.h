#pragma once

#include "boost/url/url.hpp"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct initial {
  api::initial_response operator()(boost::urls::url_view const&) const;

  nigiri::timetable const& tt_;
  config const& config_;
};

}  // namespace motis::ep