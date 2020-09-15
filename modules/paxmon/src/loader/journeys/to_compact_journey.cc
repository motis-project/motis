#include "motis/paxmon/loader/journeys/to_compact_journey.h"

#include "utl/verify.h"

#include "motis/core/access/station_access.h"

#include "motis/paxmon/loader/journeys/journey_access.h"

namespace motis::paxmon {

compact_journey to_compact_journey(journey const& j, schedule const& sched) {
  compact_journey cj;

  for_each_trip(
      j, sched,
      [&](trip const* trp, journey::stop const& from_stop,
          journey::stop const& to_stop,
          std::optional<transfer_info> const& ti) {
        auto const from_station_id =
            get_station(sched, from_stop.eva_no_)->index_;
        auto const to_station_id = get_station(sched, to_stop.eva_no_)->index_;

        auto const enter_time = unix_to_motistime(
            sched.schedule_begin_, from_stop.departure_.schedule_timestamp_);
        auto const exit_time = unix_to_motistime(
            sched.schedule_begin_, to_stop.arrival_.schedule_timestamp_);

        cj.legs_.emplace_back(journey_leg{trp, from_station_id, to_station_id,
                                          enter_time, exit_time, ti});
      });

  utl::verify(!cj.legs_.empty(), "to_compact_journey: empty journey");

  return cj;
}

}  // namespace motis::paxmon
