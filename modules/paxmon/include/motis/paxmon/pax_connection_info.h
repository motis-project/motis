#pragma once

#include <vector>

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_section_info {
  explicit pax_section_info(passenger_group* group) : group_{group} {}

  passenger_group* group_{};
  bool valid_{true};
};

struct pax_connection_info {
  std::vector<pax_section_info> section_infos_;
};

}  // namespace motis::paxmon
