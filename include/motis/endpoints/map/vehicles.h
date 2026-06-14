#pragma once

#include <memory>

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct vehicles {
  api::VehiclePositionsResponse operator()(boost::urls::url_view const&) const;

  tag_lookup const* tags_{nullptr};
  nigiri::timetable const* tt_{nullptr};
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep
