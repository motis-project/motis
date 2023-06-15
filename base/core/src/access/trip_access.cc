#include "motis/core/access/trip_access.h"

#include <algorithm>

#include "utl/verify.h"

#include "motis/string.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/error.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/extern_trip.h"

namespace motis {

trip const* get_gtfs_trip(schedule const& sched, gtfs_trip_id const& trip_id) {
  auto const it = sched.gtfs_trip_ids_.find(trip_id.trip_id_);
  utl::verify(it != end(sched.gtfs_trip_ids_), "unable to find GTFS trip {}",
              trip_id);

  auto const& trips = it->second;
  if (trip_id.start_date_.has_value()) {
    auto const trip_it =
        std::find_if(begin(trips), end(trips),
                     [&](mcd::pair<unixtime, ptr<trip const>> const& trp) {
                       return trp.first == *trip_id.start_date_;
                     });
    utl::verify(trip_it != end(trips),
                "unable to find GTFS trip {} with at date {}", trip_id,
                format_unix_time(*trip_id.start_date_));
    return trip_it->second;
  } else if (it->second.size() > 1) {
    auto const n = now();
    auto const closest_to_now =
        std::min_element(begin(it->second), end(it->second),
                         [&](mcd::pair<unixtime, ptr<trip const>> const& a,
                             mcd::pair<unixtime, ptr<trip const>> const& b) {
                           return (a.first - n) < (b.first - n);
                         });
    LOG(logging::warn) << "ambiguous trip id " << trip_id << " w/o date, using "
                       << format_unix_time(closest_to_now->first, "%F");
    return closest_to_now->second;
  } else {
    return it->second.front().second;
  }
}

trip const* get_trip(schedule const& sched, std::string_view eva_nr,
                     uint32_t const train_nr, unixtime const timestamp,
                     std::string_view target_eva_nr,
                     unixtime const target_timestamp, std::string_view line_id,
                     bool const fuzzy) {
  auto const station_id = get_station(sched, eva_nr)->index_;
  auto const motis_time = unix_to_motistime(sched, timestamp);
  auto const primary_id = primary_trip_id(station_id, train_nr, motis_time);

  auto it =
      std::lower_bound(begin(sched.trips_), end(sched.trips_),
                       std::make_pair(primary_id, static_cast<trip*>(nullptr)));
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

trip const* get_trip(schedule const& sched, extern_trip const& e_trp) {
  return get_trip(sched, e_trp.station_id_, e_trp.train_nr_, e_trp.time_,
                  e_trp.target_station_id_, e_trp.target_time_, e_trp.line_id_);
}

trip const* get_trip(schedule const& sched, trip_idx_t const idx) {
  return sched.trip_mem_.at(idx).get();
}

trip const* find_trip(schedule const& sched, primary_trip_id id) {
  auto it = std::lower_bound(begin(sched.trips_), end(sched.trips_),
                             std::make_pair(id, static_cast<trip*>(nullptr)));
  if (it != end(sched.trips_) && it->first == id) {
    return it->second;
  }
  return nullptr;
}

trip const* find_trip(schedule const& sched, full_trip_id id) {
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

unsigned stop_seq_to_stop_idx(trip const& trp, unsigned stop_seq) {
  if (trp.stop_seq_numbers_.empty() || stop_seq == 0U) {
    return stop_seq;
  } else {
    auto const it = std::lower_bound(begin(trp.stop_seq_numbers_),
                                     end(trp.stop_seq_numbers_), stop_seq);
    utl::verify(it != end(trp.stop_seq_numbers_) && *it == stop_seq,
                "stop sequence {} for trip {} not found, sequence: {}",
                stop_seq, trp.dbg_.str(), trp.stop_seq_numbers_);
    return static_cast<unsigned>(
        std::distance(begin(trp.stop_seq_numbers_), it));
  }
}

}  // namespace motis
