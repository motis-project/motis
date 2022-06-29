#include "motis/mcraptor/cpu/cpu_raptor.h"

namespace motis::mcraptor {

trip_count get_earliest_trip(raptor_timetable const& tt,
                             raptor_route const& route,
                             Bag* prev_round,
                             stop_times_index const r_stop_offset) {

  stop_id const stop_id =
      tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

  // station was never visited, there can't be a earliest trip
  if (!prev_round[stop_id].isValid()) {
    return invalid<trip_count>;
  }

  // get first defined earliest trip for the stop in the route
  auto const first_trip_stop_idx = route.index_to_stop_times_ + r_stop_offset;
  auto const last_trip_stop_idx =
      first_trip_stop_idx + ((route.trip_count_ - 1) * route.stop_count_);

  trip_count current_trip = 0;
  for (auto stop_time_idx = first_trip_stop_idx;
       stop_time_idx <= last_trip_stop_idx;
       stop_time_idx += route.stop_count_) {

    auto const stop_time = tt.stop_times_[stop_time_idx];
    if (valid(stop_time.departure_) &&
        prev_round[stop_id].getEarliestArrivalTime() <= stop_time.departure_) {
      return current_trip;
    }

    ++current_trip;
  }

  return invalid<trip_count>;
}

void init_arrivals(Rounds& result, raptor_query const& q,
                   cpu_mark_store& station_marks) {

  // Don't set the values for the earliest arrival, as the footpath update
  // in the first round will use the values in conjunction with the
  // footpath lengths without transfertime leading to invalid results.
  // Not setting the earliest arrival values should (I hope) be correct.
  Label newLabel(0, q.source_time_begin_, 0);
  result[0][q.source_].merge(newLabel);
  station_marks.mark(q.source_);

//  for (auto const& add_start : q.add_starts_) {
//    time const add_start_time = q.source_time_begin_ + add_start.offset_;
//    Label addStartLabel(0, add_start_time, 0);
//    result[0][add_start.s_id_].merge(addStartLabel);
//    station_marks.mark(add_start.s_id_);
//  }
}

void update_route(raptor_timetable const& tt, route_id const r_id,
                  Bag* prev_round, Bag* current_round,
                  BestBags& ea, cpu_mark_store& station_marks) {
  auto const& route = tt.routes_[r_id];

  trip_count earliest_trip_id = invalid<trip_count>;
  for (stop_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
       ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id =
          get_earliest_trip(tt, route, prev_round, r_stop_offset);
      continue;
    }

    auto const stop_id =
        tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
    auto const current_stop_time_idx = route.index_to_stop_times_ +
                                       (earliest_trip_id * route.stop_count_) +
                                       r_stop_offset;

    auto const& stop_time = tt.stop_times_[current_stop_time_idx];

    // need the minimum due to footpaths updating arrivals
    // and not earliest arrivals
    Label stopTimeLabel(0, stop_time.arrival_, 0);

    if (stopTimeLabel.dominatesAll(ea[stop_id].labels)
        && stopTimeLabel.dominatesAll(current_round[stop_id].labels)) {
      station_marks.mark(stop_id);
      current_round[stop_id].merge(stopTimeLabel);
    }

    /*
     * The reason for the split in the update process for the current_round
     * and the earliest arrivals is that we might have some results in
     * current_round from former runs of the algorithm, but the earliest
     * arrivals start at invalid<time> every run.
     *
     * Therefore, we need to set the earliest arrival independently from
     * the results in current round.
     *
     * We cannot carry over the earliest arrivals from former runs, since
     * then we would skip on updates to the curren_round results.
     */

    if (stopTimeLabel.dominatesAll(ea[stop_id].labels)) {
      ea[stop_id].merge(stopTimeLabel);
    }

    // check if we could catch an earlier trip
    auto const previous_k_arrival = prev_round[stop_id].getEarliestArrivalTime();
    if (previous_k_arrival <= stop_time.departure_) {
      earliest_trip_id =
          std::min(earliest_trip_id,
                   get_earliest_trip(tt, route, prev_round, r_stop_offset));
    }
  }
}

void update_footpaths(raptor_timetable const& tt, Bag* current_round,
                      BestBags& ea,
                      cpu_mark_store& station_marks) {

  for (stop_id stop_id = 0; stop_id < tt.stop_count(); ++stop_id) {

    auto index_into_transfers = tt.stops_[stop_id].index_to_transfers_;
    auto next_index_into_transfers = tt.stops_[stop_id + 1].index_to_transfers_;

    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {

      auto const& footpath = tt.footpaths_[current_index];

      if (!ea[stop_id].isValid()) {
        continue;
      }

      // there is no triangle inequality in the footpath graph!
      // we cannot use the normal arrival values,
      // but need to use the earliest arrival values as read
      // and write to the normal arrivals,
      // otherwise it is possible that two footpaths
      // are chained together
      time const new_arrival = ea[stop_id].getEarliestArrivalTime() + footpath.duration_;

      Bag to_earliest_arrival = ea[footpath.to_];
      Bag to_arrival = current_round[footpath.to_];

      Label newArrivalLabel(0, new_arrival, 0);

      if (newArrivalLabel.dominatesAll(to_earliest_arrival.labels)
          && newArrivalLabel.dominatesAll(to_arrival.labels)) {
        station_marks.mark(footpath.to_);
        current_round[footpath.to_].merge(newArrivalLabel);
      }
    }
  }
}

void invoke_cpu_raptor(raptor_query const& query, raptor_statistics&) {
  auto const& tt = query.tt_;

  Rounds& result = query.result();

  BestBags ea;
  for(int i = 0; i < tt.stop_count(); i++) {
    ea.push_back(Bag());
  }

  cpu_mark_store station_marks(tt.stop_count());
  cpu_mark_store route_marks(tt.route_count());

  init_arrivals(result, query, station_marks);

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    bool any_marked = false;

    for (auto s_id = 0; s_id < tt.stop_count(); ++s_id) {

      if (!station_marks.marked(s_id)) {
        continue;
      }

      if (!any_marked) {
        any_marked = true;
      }

      auto const& stop = tt.stops_[s_id];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        route_marks.mark(tt.stop_routes_[sri]);
      }
    }

    if (!any_marked) {
      break;
    }

    station_marks.reset();

    for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
      if (!route_marks.marked(r_id)) {
        continue;
      }

      update_route(tt, r_id, result[round_k - 1], result[round_k], ea,
                   station_marks);
    }

    route_marks.reset();

    update_footpaths(tt, result[round_k], ea, station_marks);
  }
}

}  // namespace motis::mcraptor
