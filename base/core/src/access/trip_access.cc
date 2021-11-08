#include "motis/core/access/trip_access.h"

#include <algorithm>

#include "motis/string.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/error.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/extern_trip.h"

namespace motis {

trip_info const* get_trip(schedule const& sched, std::string const& trip_id,
                          std::time_t const date) {
  if (auto it = sched.gtfs_trip_ids_.find({trip_id, date});
      it == end(sched.gtfs_trip_ids_) || it->first.trip_id_ != trip_id ||
      it->first.start_date_ != date) {
    throw std::runtime_error{
        "Could not find trip for the given trip id and day!"};
  } else {
    return it->second;
  }
}

trip_info const* get_trip(schedule const& sched, std::string_view eva_nr,
                          uint32_t const train_nr, std::time_t const timestamp,
                          std::string_view target_eva_nr,
                          std::time_t const target_timestamp,
                          std::string_view line_id, bool const fuzzy) {
  auto const station_id = get_station(sched, eva_nr)->index_;
  auto const motis_time = unix_to_motistime(sched, timestamp);
  auto const primary_id = primary_trip_id{station_id, train_nr, motis_time};

  auto it = std::lower_bound(
      begin(sched.trips_), end(sched.trips_),
      std::make_pair(primary_id, static_cast<trip_info*>(nullptr)));
  if (it == end(sched.trips_) || !(it->first == primary_id)) {
    throw std::system_error(access::error::service_not_found);
  }

  auto const target_station_id = get_station(sched, target_eva_nr)->index_;
  auto const target_motis_time =
      unix_to_motistime(sched.schedule_begin_, target_timestamp);
  for (; it != end(sched.trips_) && it->first == primary_id; ++it) {
    auto const& s = it->second->id_.secondary_;
    if ((fuzzy || line_id == s.line_id_) &&
        target_station_id == s.target_station_id_ &&
        target_motis_time == s.target_time_) {
      return it->second;
    }
  }

  throw std::system_error(access::error::service_not_found);
}

trip_info const* get_trip(schedule const& sched, extern_trip const& e_trp) {
  return get_trip(sched, e_trp.station_id_, e_trp.train_nr_, e_trp.time_,
                  e_trp.target_station_id_, e_trp.target_time_, e_trp.line_id_);
}

trip_info const* find_trip(schedule const& sched, primary_trip_id id) {
  auto it = std::lower_bound(begin(sched.trips_), end(sched.trips_),
                             std::make_pair(id, static_cast<trip*>(nullptr)));
  if (it != end(sched.trips_) && it->first == id) {
    return it->second;
  }
  return nullptr;
}

trip_info const* find_trip(schedule const& sched, full_trip_id id) {
  for (auto it = std::lower_bound(
           begin(sched.trips_), end(sched.trips_),
           std::make_pair(id.primary_, static_cast<trip*>(nullptr)));
       it != end(sched.trips_) && it->first == id.primary_; ++it) {
    if (it->second->id_.secondary_ == id.secondary_) {
      return it->second;
    }
  }
  return nullptr;
}

}  // namespace motis
