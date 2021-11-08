#include "motis/revise/get_ev_key.h"

#include <algorithm>

#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"

namespace motis::revise {

ev_key get_ev_key(schedule const& sched, journey const& j,
                  unsigned const stop_idx, event_type const ev_type) {
  auto const is_arr = (ev_type == event_type::ARR);
  auto const journey_trp =
      std::find_if(begin(j.trips_), end(j.trips_), [&](journey::trip const& t) {
        return is_arr ? (t.to_ >= stop_idx && t.from_ < stop_idx)
                      : (t.to_ > stop_idx && t.from_ <= stop_idx);
      });
  utl::verify(journey_trp != end(j.trips_),
              "get ev key(trip): invalid journey");

  auto const trp = get_trip(sched, journey_trp->extern_trip_);
  auto const& stop = j.stops_[stop_idx];
  auto const schedule_time = is_arr ? stop.arrival_.schedule_timestamp_
                                    : stop.departure_.schedule_timestamp_;
  return get_ev_key_from_trip(sched, trp, stop.eva_no_, ev_type, schedule_time);
}

ev_key get_ev_key_from_trip(schedule const& sched, trip const* trp,
                            std::string const& station_id,
                            event_type const& ev_type,
                            unixtime const schedule_time) {
  auto const is_arr = (ev_type == event_type::ARR);
  auto const s_node = get_station_node(sched, station_id);
  utl::verify(trp != nullptr, "invalid trip");

  auto const re_it = std::find_if(
      begin(*trp->edges_), end(*trp->edges_), [&](trip_info::route_edge const& re) {
        auto const ev = ev_key{re, trp->lcon_idx_, ev_type};
        return (is_arr ? re->to_ : re->from_)->station_node_ == s_node &&
               motis_to_unixtime(sched.schedule_begin_,
                                 get_schedule_time(sched, ev)) == schedule_time;
      });
  return re_it != end(*trp->edges_) ? ev_key{*re_it, trp->lcon_idx_, ev_type}
                                    : ev_key{};
}

}  // namespace motis::revise
