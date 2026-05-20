#pragma once

#include "boost/url/url.hpp"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

api::initial_response get_initial_response(data const&,
                                           std::string_view motis_version);

struct initial {
  api::initial_response operator()(boost::urls::url_view const&) const;

  api::initial_response const& response_;
};

}  // namespace motis::ep
