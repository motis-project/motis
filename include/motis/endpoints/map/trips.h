#pragma once

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct trips {
  api::trips_response operator()(boost::urls::url_view const&) const;

  tag_lookup const& tags_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  railviz_static_index const& static_;
};

}  // namespace motis::ep