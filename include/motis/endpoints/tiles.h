#pragma once

#include "boost/url/url_view.hpp"

#include "motis/fwd.h"
#include "motis/http_response.h"

namespace motis::ep {

struct tiles {
  http_response operator()(boost::urls::url_view const&) const;

  tiles_data& tiles_data_;
};

}  // namespace motis::ep