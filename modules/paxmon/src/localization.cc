#include "motis/paxmon/localization.h"

#include <algorithm>
#include <vector>

#include "utl/to_vec.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

namespace motis::paxmon {

passenger_localization localize(schedule const& sched,
                                reachability_info const& reachability,
                                time const localization_time) {
  std::vector<std::uint32_t> remaining_interchanges;
  for (auto tripit = std::rbegin(reachability.reachable_trips_);
       tripit != std::rend(reachability.reachable_trips_); ++tripit) {
    if (localization_time <= tripit->enter_real_time_) {
      // passenger has not yet entered this trip
      remaining_interchanges.emplace_back(tripit->leg_->exit_station_id_);
      continue;
    }
    if (tripit->valid_exit() && localization_time > tripit->exit_real_time_) {
      // passenger has already exited this trip
      return {nullptr,
              sched.stations_[tripit->leg_->exit_station_id_].get(),
              tripit->exit_schedule_time_,
              tripit->exit_real_time_,
              false,
              std::move(remaining_interchanges)};
    }
    // passenger is currently in this trip
    auto const* trp = get_trip(sched, tripit->trip_idx_);
    auto sections = access::sections(trp);
    auto lb =
        std::lower_bound(begin(sections), end(sections), localization_time,
                         [](access::trip_section const& section, auto const t) {
                           return section.lcon().d_time_ < t;
                         });
    if (lb == end(sections) || (*lb).lcon().d_time_ != localization_time) {
      lb--;
    }
    remaining_interchanges.emplace_back(tripit->leg_->exit_station_id_);
    auto const current_section = *lb;
    return {trp,
            sched.stations_[current_section.to_station_id()].get(),
            get_schedule_time(sched, current_section.ev_key_to()),
            current_section.lcon().a_time_,
            false,
            std::move(remaining_interchanges)};
  }
  // passenger has not yet entered any trips
  if (!reachability.reachable_interchange_stations_.empty()) {
    // passenger is at the first station
    auto const& reachable_station =
        reachability.reachable_interchange_stations_.front();
    return {nullptr,
            sched.stations_[reachable_station.station_].get(),
            reachable_station.schedule_time_,
            reachable_station.real_time_,
            true,
            utl::to_vec(reachability.reachable_interchange_stations_,
                        [](auto const& rs) { return rs.station_; })};
  }
  // shouldn't happen
  return {};
}

}  // namespace motis::paxmon
