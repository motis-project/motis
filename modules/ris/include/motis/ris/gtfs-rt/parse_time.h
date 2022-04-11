#pragma once

#include <ctime>

#include "boost/date_time/gregorian/gregorian_types.hpp"

#include "utl/verify.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event.h"
#include "motis/core/access/realtime_access.h"

#include "gtfsrt.pb.h"

namespace motis {
struct schedule;
struct trip;
namespace ris::gtfsrt {

inline std::optional<unixtime> parse_start_date(std::string const& yyyymmdd) {
  utl::verify(yyyymmdd.empty() || yyyymmdd.length() == 8,
              "start date expected to be empty or YYYYMMDD, is \"{}\"",
              yyyymmdd);
  if (yyyymmdd.empty()) {
    return std::nullopt;
  } else {
    return to_unix_time(std::stoi(yyyymmdd.substr(0, 4)),
                        std::stoi(yyyymmdd.substr(4, 2)),
                        std::stoi(yyyymmdd.substr(6, 2)));
  }
}

inline gtfs_trip_id to_trip_id(transit_realtime::TripDescriptor const& d,
                               std::string const& tag) {
  return {tag, d.trip_id(), parse_start_date(d.start_date())};
}

inline unixtime get_updated_time(
    transit_realtime::TripUpdate_StopTimeEvent const& time_event,
    unixtime schedule_time, const bool is_addition_trip) {

  unixtime updated_time = 0;
  if (time_event.has_time() && !is_addition_trip) {
    updated_time = time_event.time();
  } else if (time_event.has_delay() && schedule_time != 0) {
    updated_time = schedule_time + time_event.delay();
  } else {
    if (is_addition_trip) {
      throw std::runtime_error{
          "get_update_time called for addition trip without delay; This should "
          "not happen!"};
    } else {
      throw std::runtime_error{
          "Neither absolute new time or schedule time and delay are given."};
    }
  }

  return updated_time;
};

inline unixtime get_schedule_time(trip const& trip, schedule const& sched,
                                  const int stop_idx, const event_type type) {
  auto const edge_idx = type == event_type::DEP ? stop_idx : stop_idx - 1;
  auto const orig_ev_key =
      get_orig_ev_key(sched, {trip.edges_->at(edge_idx), trip.lcon_idx_, type});
  auto const orig_sched_time = get_schedule_time(sched, orig_ev_key);
  return motis_to_unixtime(sched.schedule_begin_, orig_sched_time);
};

}  // namespace ris::gtfsrt
}  // namespace motis
