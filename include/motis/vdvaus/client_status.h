#pragma once

#include <string>

#include "boost/url/url_view.hpp"

#include "motis/fwd.h"

namespace motis::vdvaus {

struct client_status {
  std::string operator()(std::string_view) const;

  connection const& vdvaus_;
};

}  // namespace motis::vdvaus