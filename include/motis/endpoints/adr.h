#pragma once

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"

namespace motis::ep {

struct adr {
  api::geocode_response operator()(boost::urls::url_view const& url) const;

  ::adr::typeahead const& t_;
  ::adr::cache& cache_;
};

}  // namespace motis::ep