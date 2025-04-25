#pragma once

#include <string>
#include <string_view>

namespace motis::vdvaus {

struct data_ready {
  std::string operator()(std::string_view) const;
};

}  // namespace motis::vdvaus