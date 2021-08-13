#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

const passenger_group* paxmon_data::get_passenger_group(
    std::uint64_t id) const {
  return graph_.passenger_groups_.at(id);
}

}  // namespace motis::paxmon
