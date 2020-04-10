#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/trip_access.h"

#include "motis/protocol/RISMessage_generated.h"

#include "motis/rt/statistics.h"

namespace motis::rt {

inline trip const* find_trip_fuzzy(statistics& stats, schedule const& sched,
                                   ris::IdEvent const* id) {
  ++stats.trip_total_;

  auto const station = find_station(sched, id->station_id()->str());
  if (station == nullptr) {
    ++stats.trip_station_not_found_;
    return nullptr;
  }

  auto const motis_time = unix_to_motistime(sched, id->schedule_time());
  if (motis_time == INVALID_TIME) {
    ++stats.trip_time_not_found_;
    return nullptr;
  }

  auto trp = find_trip(
      sched, primary_trip_id{station->index_, id->service_num(), motis_time});
  if (trp != nullptr) {
    return trp;
  }
  ++stats.trip_primary_not_found_;

  trp = find_trip(sched, primary_trip_id{station->index_, 0, motis_time});
  if (trp != nullptr) {
    return trp;
  }
  ++stats.trip_primary_0_not_found_;

  return nullptr;
}

}  // namespace motis::rt
