#pragma once

#include <memory>

#include "boost/url/url_view.hpp"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::ep {

struct rental {
  api::rentals_response operator()(boost::urls::url_view const&) const;

  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
};

}  // namespace motis::ep
