#include "motis/ris/gtfs-rt/parse_stop.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_stop.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/parse_time.h"

using namespace transit_realtime;

namespace motis::ris::gtfsrt {

int get_stop_edge_idx(const int stop_idx, const event_type type) {
  return type == event_type::DEP ? stop_idx : stop_idx - 1;
}

std::string parse_stop_id(std::string const& stop_id) {
  auto colon_idx = stop_id.find_first_of(':');
  return colon_idx != std::string::npos ? stop_id.substr(0, colon_idx)
                                        : stop_id;
}

int get_future_stop_idx(trip const& trip, schedule& sched,
                        const int last_stop_idx, std::string const& stop_id) {
  access::stops stops(&trip);
  auto idx_offset =
      last_stop_idx != std::numeric_limits<int>::max() ? last_stop_idx : 0;
  auto trip_stop = std::find_if(begin(stops) + idx_offset, end(stops),
                                [&](access::trip_stop const& stop) {
                                  auto station = stop.get_station(sched);
                                  return station.eva_nr_ == stop_id;
                                });
  if (trip_stop == end(stops) ||
      stop_id != (*trip_stop).get_station(sched).eva_nr_) {
    throw std::runtime_error("Could not find stop " + stop_id);
  }
  return (*trip_stop).index();
}

void update_stop_idx(stop_context& current_stop, schedule& sched,
                     trip const& trip,
                     TripUpdate_StopTimeUpdate const& stop_time_upd,
                     known_stop_skips* stop_skips) {
  // Method to obtain the current stop idx
  // A. is the stop idx for the given stop time update already known?
  // check for same sequence number; if this is not given GTFS-RT
  // requires the stop id to be unique along the route
  // it is therefore used as fallback here
  auto const has_sequ = stop_time_upd.has_stop_sequence();
  auto const has_stop = stop_time_upd.has_stop_id();
  if ((has_sequ && stop_time_upd.stop_sequence() == current_stop.seq_no_) ||
      (has_stop && stop_time_upd.stop_id() == current_stop.station_id_)) {
    return;
  }
  current_stop.is_skip_known_ = false;

  // B. Okay the sequence number or stop id has changed -> determine the new
  // stop idx
  // as the stop sequence is not needed to be a list of consecutive numbers
  // diverse strategies are applied to get the correct index of the current
  // stop on hand fast
  // 1. best case is the number is consecutive
  // 2. if not it may be consecutive in parts
  // 3. or it might be just randomly increasing
  if (has_sequ &&
      current_stop.seq_no_ !=
          std::numeric_limits<decltype(current_stop.seq_no_)>::max() &&
      current_stop.seq_no_ + 1 == stop_time_upd.stop_sequence() &&
      current_stop.idx_ != std::numeric_limits<int>::max()) {
    // nicely consecutive
    current_stop.seq_no_ = stop_time_upd.stop_sequence();
    if (stop_skips == nullptr ||
        !stop_skips->is_skipped(current_stop.seq_no_)) {

      current_stop.idx_ = current_stop.idx_ + 1;
      if (has_stop) {
        current_stop.station_id_ = stop_time_upd.stop_id();
      } else if (has_sequ && current_stop.idx_ >= 0) {
        current_stop.station_id_ = access::trip_stop{&trip, current_stop.idx_}
                                       .get_station(sched)
                                       .eva_nr_;
      } else {
        throw std::runtime_error{
            "Neither stop, nor sequence id given. No valid GTFS-RT!"};
      }

    } else {
      current_stop.is_skip_known_ = true;
    }
  } else {
    // okay. numbers are not consecutive. this can have two reasons:
    // 1. there is no StopTime Update entry for this stop or
    // 2. the numbers increase not consecutively along the route
    // whichever situation does not matter a search is required
    if (has_stop) {
      current_stop.station_id_ = stop_time_upd.stop_id();
      current_stop.seq_no_ = has_sequ ? stop_time_upd.stop_sequence()
                                      : std::numeric_limits<int>::max();
      if (!has_sequ || stop_skips == nullptr ||
          !stop_skips->is_skipped(current_stop.seq_no_)) {
        current_stop.idx_ = get_future_stop_idx(trip, sched, current_stop.idx_,
                                                stop_time_upd.stop_id());
      } else {
        current_stop.is_skip_known_ = true;
      }
    } else {
      // it is currently not possible to obtain the
      // stop idx just by having the sequence number
      // therfore error out
      throw std::runtime_error{
          "Unable to obtain stop idx just by non consecutive sequence numbers"};
    }
  }
};

void stop_context::update(schedule& sched, trip const& trip,
                          TripUpdate_StopTimeUpdate const& stu,
                          known_stop_skips* stop_skips) {
  update_stop_idx(*this, sched, trip, stu, stop_skips);
  if (!is_skip_known_ && idx_ > 0) {
    stop_arrival_ = get_schedule_time(trip, sched, idx_, event_type::ARR);
  }

  if (!is_skip_known_ &&
      get_stop_edge_idx(idx_, event_type::DEP) < trip.edges_->size()) {
    stop_departure_ = get_schedule_time(trip, sched, idx_, event_type::DEP);
  }
}

}  // namespace motis::ris::gtfsrt