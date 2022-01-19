#include "motis/ris/gtfs-rt/parse_stop.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_stop.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/parse_time.h"

using namespace transit_realtime;

namespace motis::ris::gtfsrt {

int get_stop_edge_idx(int const stop_idx, const event_type type) {
  return type == event_type::DEP ? stop_idx : stop_idx - 1;
}

std::string parse_stop_id(std::string const& stop_id) { return stop_id; }

void update_stop_idx(stop_context& current_stop, schedule const& sched,
                     trip const& trip,
                     TripUpdate_StopTimeUpdate const& stop_time_upd,
                     known_stop_skips* stop_skips, std::string const& tag) {
  if (stop_time_upd.has_stop_sequence() &&
      stop_time_upd.stop_sequence() == current_stop.seq_no_ &&
      stop_time_upd.has_stop_id() &&
      tag + stop_time_upd.stop_id() == current_stop.station_id_) {
    return;
  }

  if (stop_time_upd.has_stop_sequence()) {
    current_stop.seq_no_ = stop_time_upd.stop_sequence();
  }

  current_stop.is_skip_known_ = stop_skips != nullptr &&
                                stop_time_upd.has_stop_sequence() &&
                                stop_skips->is_skipped(current_stop.seq_no_);
  if (current_stop.is_skip_known_) {
    return;
  }

  if (stop_time_upd.has_stop_sequence()) {
    auto const stop_idx =
        stop_seq_to_stop_idx(trip, stop_time_upd.stop_sequence());
    auto const station_idx =
        access::trip_stop{&trip, static_cast<int>(stop_idx)}
            .get_station(sched)
            .index_;
    current_stop.station_id_ = sched.stations_.at(station_idx)->eva_nr_;
    current_stop.idx_ = stop_idx;
  } else if (stop_time_upd.has_stop_id()) {
    current_stop.station_id_ = tag + stop_time_upd.stop_id();

    auto const stops = access::stops(&trip);
    auto const it = std::find_if(
        begin(stops), end(stops), [&](access::trip_stop const& stop) {
          return stop.get_station(sched).eva_nr_ == current_stop.station_id_;
        });
    utl::verify(it != end(stops), "trip {} has no station {}", trip.dbg_,
                current_stop.station_id_);
    current_stop.idx_ = static_cast<int>(std::distance(begin(stops), it));
  } else {
    throw utl::fail("update for trip {} w/o station and seq", trip.dbg_);
  }
};

void stop_context::update(schedule const& sched, trip const& trip,
                          TripUpdate_StopTimeUpdate const& stu,
                          known_stop_skips* stop_skips,
                          std::string const& tag) {
  update_stop_idx(*this, sched, trip, stu, stop_skips, tag);
  if (!is_skip_known_ && idx_ > 0) {
    stop_arrival_ = get_schedule_time(trip, sched, idx_, event_type::ARR);
  }

  if (!is_skip_known_ &&
      get_stop_edge_idx(idx_, event_type::DEP) < trip.edges_->size()) {
    stop_departure_ = get_schedule_time(trip, sched, idx_, event_type::DEP);
  }
}

}  // namespace motis::ris::gtfsrt
