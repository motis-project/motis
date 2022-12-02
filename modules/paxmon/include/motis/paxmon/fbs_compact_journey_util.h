#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/station_access.h"

#include "motis/protocol/PaxMonCompactJourney_generated.h"

namespace motis::paxmon {

inline unsigned get_destination_station_id(schedule const& sched,
                                           PaxMonCompactJourney const* cj) {
  utl::verify(cj->legs()->size() > 0,
              "paxmon::get_destination_station_id: empty journey");
  auto const* fbs_station =
      cj->legs()->Get(cj->legs()->size() - 1)->exit_station();
  return get_station(sched, fbs_station->id()->str())->index_;
}

}  // namespace motis::paxmon
