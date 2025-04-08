#pragma once

#include <string>

#include "boost/url/url_view.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

struct client_status {
  std::string operator()(boost::urls::url_view const&) const;

  vdv_rt::connection const* con_;
};

}  // namespace motis::vdv_rt