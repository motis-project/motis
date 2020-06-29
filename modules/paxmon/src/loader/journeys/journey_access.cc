#include "motis/paxmon/loader/journeys/journey_access.h"

#include <stdexcept>

#include "utl/enumerate.h"

#include "motis/core/access/station_access.h"

namespace motis::paxmon {

journey::trip const* get_journey_trip(journey const& j,
                                      std::size_t enter_stop_idx) {
  journey::trip const* jt = nullptr;
  std::size_t exit_stop_idx = 0;
  for (auto const& t : j.trips_) {
    if (t.from_ == enter_stop_idx && t.to_ > exit_stop_idx) {
      jt = &t;
      exit_stop_idx = t.to_;
    }
  }
  return jt;
}

void for_each_trip(
    journey const& j, schedule const& sched,
    std::function<void(extern_trip const&, journey::stop const&,
                       journey::stop const&,
                       std::optional<transfer_info> const&)> const& cb) {
  std::optional<std::size_t> exit_stop_idx;
  for (auto const& [stop_idx, stop] : utl::enumerate(j.stops_)) {
    if (stop.exit_) {
      exit_stop_idx = stop_idx;
    }
    if (stop.enter_) {
      auto const jt = get_journey_trip(j, stop_idx);
      if (jt == nullptr) {
        throw std::runtime_error{"invalid journey: trip not found"};
      }
      std::optional<transfer_info> transfer;
      if (exit_stop_idx) {
        if (*exit_stop_idx == stop_idx) {
          auto const st = get_station(sched, stop.eva_no_);
          transfer = transfer_info{static_cast<duration>(st->transfer_time_),
                                   transfer_info::type::SAME_STATION};
        } else {
          auto const walk_duration =
              (stop.arrival_.schedule_timestamp_ -
               j.stops_.at(*exit_stop_idx).departure_.schedule_timestamp_) /
              60;
          transfer = transfer_info{static_cast<duration>(walk_duration),
                                   transfer_info::type::FOOTPATH};
        }
      }
      cb(jt->extern_trip_, stop, j.stops_.at(jt->to_), transfer);
      exit_stop_idx.reset();
    }
  }
}

}  // namespace motis::paxmon
