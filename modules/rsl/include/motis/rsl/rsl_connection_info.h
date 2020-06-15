#pragma once

#include <vector>

#include "motis/rsl/passenger_group.h"

namespace motis::rsl {

struct pax_section_info {
  explicit pax_section_info(passenger_group* group) : group_{group} {}

  passenger_group* group_{};
  bool valid_{true};
};

struct rsl_connection_info {
  std::vector<pax_section_info> section_infos_;
};

}  // namespace motis::rsl
