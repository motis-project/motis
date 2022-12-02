#pragma once

#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::measures {

struct affected_groups_info {
  mcd::hash_map<paxmon::passenger_group_with_route,
                paxmon::passenger_localization>
      localization_;
  mcd::hash_map<paxmon::passenger_group_with_route,
                mcd::vector<measure_variant const*>>
      measures_;
};

affected_groups_info get_affected_groups(schedule const& sched,
                                         motis::paxmon::universe& uv,
                                         time loc_time,
                                         measure_set const& measures);

}  // namespace motis::paxforecast::measures
