#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

const passenger_group* paxmon_data::get_passenger_group(
    passenger_group_index id) const {
  return graph_.passenger_groups_.at(id);
}

}  // namespace motis::paxmon
