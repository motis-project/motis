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

concrete_trip get_trip(schedule const& sched, gtfs_trip_id const& trip_id) {
  auto const get_set_bit = [&](bitfield const& b, day_idx_t const start = 0U) {
    for (auto i = start; i < MAX_DAYS; ++i) {
      if (b.test(i)) {
        return i;
      }
    }
    return day_idx_t{-1};
  };

  if (auto it = sched.gtfs_trip_ids_.find(trip_id.trip_id_);
      it == end(sched.gtfs_trip_ids_)) {
    throw std::runtime_error{
        "Could not find trip for the given trip id and day!"};
  } else if (trip_id.start_date_.has_value()) {
    // start day set - check if service operates
    auto const start_day_idx =
        unix_to_motistime(sched, *trip_id.start_date_).day();
    utl::verify(it->second->operates_on_day(start_day_idx),
                "trip {} does not operate on day {}",
                format_unix_time(*trip_id.start_date_, "%F"));
    return {it->second, start_day_idx};
  } else if (it->second->ctrp_count() == 1U) {
    // unambiguous day
    return {it->second, get_set_bit(it->second->traffic_days())};
  } else {
    // ambiguous day - closest to today
    auto const n = now();
    auto const day_idx = get_set_bit(it->second->traffic_days(),
                                     unix_to_motistime(sched, n).day());
    utl::verify(day_idx != -1, "no traffic day for {} after {} found",
                trip_id.trip_id_, format_unix_time(n, "%F"));
    return {it->second, day_idx};
  }
}

concrete_trip get_trip(schedule const& sched, std::string_view eva_nr,
                       uint32_t const train_nr, unixtime const timestamp,
                       std::string_view target_eva_nr,
                       unixtime const target_timestamp,
                       std::string_view line_id, bool const fuzzy) {
  auto const station_id = get_station(sched, eva_nr)->index_;
  auto const first_departure_mam = unix_to_motistime(sched, timestamp).mam();
  auto const primary_id =
      primary_trip_id{station_id, train_nr, first_departure_mam};

  auto it = std::lower_bound(
      begin(sched.trips_), end(sched.trips_),
      std::make_pair(primary_id, static_cast<trip_info*>(nullptr)));
  if (it == end(sched.trips_) || !(it->first == primary_id)) {
    std::cerr << "sizeof=" << sizeof(primary_trip_id) << "\n";
    std::cerr << "NEEDLE: " << primary_id << "\n";
    for (auto const& [id, trp] : sched.trips_) {
      std::cerr << "  available: " << id << "\n";
    }

    throw std::system_error(access::error::service_not_found);
  }

  auto const day_idx = unix_to_motistime(sched, timestamp).day();
  auto const target_station_id = get_station(sched, target_eva_nr)->index_;
  auto const last_arrival_mam =
      unix_to_motistime(sched.schedule_begin_, target_timestamp).mam();
  for (; it != end(sched.trips_) && it->first == primary_id; ++it) {
    auto const& s = it->second->id_.secondary_;
    if ((fuzzy || line_id == s.line_id_) &&
        target_station_id == s.target_station_id_ &&
        last_arrival_mam == s.last_arrival_mam_ &&
        it->second->edges_->front()->operates_on_day(day_idx)) {
      return {it->second, day_idx};
    }
  }

  throw std::system_error(access::error::service_not_found);
}

concrete_trip get_trip(schedule const& sched, extern_trip const& e_trp) {
  return get_trip(sched, e_trp.station_id_, e_trp.train_nr_, e_trp.time_,
                  e_trp.target_station_id_, e_trp.target_time_, e_trp.line_id_);
}

std::optional<concrete_trip> find_trip(schedule const& sched,
                                       primary_trip_id const& id,
                                       day_idx_t const day_idx) {
  auto it =
      std::lower_bound(begin(sched.trips_), end(sched.trips_),
                       std::make_pair(id, static_cast<trip_info*>(nullptr)));
  for (; it != end(sched.trips_) && it->first == id; ++it) {
    if (it->second->edges_->front()->operates_on_day(day_idx)) {
      return concrete_trip{it->second, day_idx};
    }
  }
  return std::nullopt;
}

std::optional<concrete_trip> find_trip(schedule const& sched,
                                       full_trip_id const& id,
                                       day_idx_t const day_idx) {
  for (auto it = std::lower_bound(
           begin(sched.trips_), end(sched.trips_),
           std::make_pair(id.primary_, static_cast<trip_info*>(nullptr)));
       it != end(sched.trips_) && it->first == id.primary_; ++it) {
    if (it->second->id_.secondary_ == id.secondary_ &&
        it->second->edges_->front()->operates_on_day(day_idx)) {
      return concrete_trip{it->second, day_idx};
    }
  }
  return std::nullopt;
}

unsigned stop_seq_to_stop_idx(trip_info const& trp, unsigned stop_seq) {
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
