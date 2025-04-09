#pragma once

#include <string>
#include <string_view>

namespace motis::vdv_rt {

struct data_ready {
  std::string operator()(std::string_view) const;
};

}  // namespace motis::vdv_rt