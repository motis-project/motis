#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct reverse_geocode {
  api::reverseGeocode_response operator()(
      boost::urls::url_view const& url) const;

  nigiri::timetable const& tt_;
  adr::typeahead const& t_;
  adr::reverse const& r_;
};

}  // namespace motis::ep