#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/station_access.h"

#include "motis/protocol/PaxMonCompactJourney_generated.h"

namespace motis::paxmon {

inline unsigned get_destination_station_id(schedule const& sched,
                                           PaxMonCompactJourney const* cj) {
  auto const has_final_footpath = cj->final_footpath()->size() == 1;
  utl::verify(cj->legs()->size() > 0 || has_final_footpath,
              "paxmon::get_destination_station_id: empty journey");
  auto const* fbs_station =
      has_final_footpath
          ? cj->final_footpath()->Get(0)->to_station()
          : cj->legs()->Get(cj->legs()->size() - 1)->exit_station();
  return get_station(sched, fbs_station->id()->str())->index_;
}

}  // namespace motis::paxmon
