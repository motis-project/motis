#pragma once

#include <string>

#include "boost/url/url_view.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

struct data_ready {
  std::string operator()(std::string_view) const;
};

}  // namespace motis::vdv_rt