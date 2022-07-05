#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/Bag.h"
#include <map>

namespace motis::mcraptor {

// TODO: implement mcRaptor

void McRaptor::init_arrivals() {
  startNewRound();
  Label newLabel(0, query.source_time_begin_, 0);
  arrival_by_route(query.source_, newLabel);
  startNewRound();
}

void McRaptor::arrival_by_route(stop_id stop, Label& newLabel) {
  // ??? checking for empty
  // check if this label may be dominated by other existing labels
  if(transferLabels[query.target_].dominates(newLabel)) {
    return;
  }
  if(!routeLabels[stop].merge(newLabel)) {
    return;
  }
  // add indominated label to the bags
  transferLabels[stop].merge(newLabel);
  currentRound()[stop].mergeUndominated(newLabel);
  // mark the station
  stopsForRoutes.mark(stop);
}

void McRaptor::arrival_by_transfer(stop_id stop, Label& newLabel) {
  // checking for empty??
  // check if this label may be dominated by other existing labels
  if(transferLabels[query.target_].dominates(newLabel)) {
    return;
  }
  if(!transferLabels[stop].merge(newLabel)) {
    return;
  }
  // addd indominated label to the bag
  currentRound()[stop].mergeUndominated(newLabel);
  // mark current station
  stopsForTransfers.mark(stop);
}

void McRaptor::relax_transfers() {
  stopsForTransfers.reset();
  routesServingUpdatedStops.clear();
  // iterate through all station and find marked
  for(stop_id stop = 0; stop < query.tt_.stop_count(); stop++) {
    if(!stopsForRoutes.marked(stop)) {
      continue;
    }
    stopsForTransfers.mark(stop);
    Bag& bag = previousRound()[stop];
    for(size_t i = 0; i < bag.size(); i++) {
      currentRound()[stop][i] = Label(bag[i], stop, i);
    }
  }

  for(stop_id stop = 0; stop < query.tt_.stop_count(); stop++) {
    if (!stopsForRoutes.marked(stop)) {
      continue;
    }
    Bag& bag = previousRound()[stop];
    // iterate through footpaths - coming from the update_footpath()
    // TODO: check for correctness
    auto index_into_transfers = query.tt_.stops_[stop].index_to_transfers_;
    auto next_index_into_transfers = query.tt_.stops_[stop + 1].index_to_transfers_;
    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {
      auto const& to_stop = query.tt_.footpaths_[current_index].to_;
      auto const& duration = query.tt_.footpaths_[current_index].duration_;
      for(size_t i = 0; i < bag.size(); i++) {
        Label newLabel;
        newLabel.arrivalTime = bag[i].arrivalTime + duration;
        newLabel.parentStation = stop;
        newLabel.parentIndex = i;
        newLabel.parentDepartureTime = bag[i].arrivalTime;
        arrival_by_transfer(to_stop, newLabel);
      }
    }
  }
}

void McRaptor::collect_routes_serving_updated_stops() {
  // find marked stations
  for(stop_id stop = 0; stop < query.tt_.stop_count(); stop++) {
    if (!stopsForTransfers.marked(stop)) {
      continue;
    }
    // find the offset in form of stop_id where the route stops by current station (getRoutesTimeForStops method)
    for(std::pair<route_id , route_stops_index> s : getRoutesTimesForStop(stop)) {
      // id of this route
      const route_id routeId = s.first;
      // route object itself
      const raptor_route route = query.tt_.routes_[routeId];
      // this offset where the route stops on this station
      const route_stops_index stop_offset = s.second;
      // if it is the last station
      if(stop_offset == route.index_to_route_stops_ + route.stop_count_ - 1) {
        continue;
      }
      // if route with this id is already in map
      // write to this route the earliest stop from both
      if(routesServingUpdatedStops.count(routeId)) {
        routesServingUpdatedStops[routeId] = std::min(routesServingUpdatedStops[routeId], stop_offset);
      } else {
        // else just add it into map
        routesServingUpdatedStops.insert(std::make_pair(routeId, stop_offset));
      }
    }
  }
}

void McRaptor::scan_routes() {
  stopsForRoutes.reset();
  for(auto i = routesServingUpdatedStops.begin(); i != routesServingUpdatedStops.end(); i++) {
    route_id routeId = i -> first;
    route_stops_index stopOffset = i -> second;
    raptor_route route = query.tt_.routes_[routeId];
    auto routeSize = route.stop_count_;

    route_stops_index routeStopsBegin = route.index_to_route_stops_;
    stop_id stop = query.tt_.route_stops_[stopOffset];

    // event represents the arrival and departure times of route in its single station
    const stop_time* firstTrip = &query.tt_.stop_times_[route.index_to_stop_times_];
    const stop_time* lastTrip = &query.tt_.stop_times_[route.index_to_stop_times_ + route.stop_count_ - 1];

    RouteBag newRouteBag;

    const auto tripSize = routeStopsBegin + routeSize;

    while(stopOffset < tripSize) {
      for(size_t j = 0; j < previousRound()[stop].size(); j++) {
        const Label& label = previousRound()[stop][j];
        const stop_time* trip = firstTrip;
        while((trip < lastTrip) && (trip[stopOffset].departure_ < label.arrivalTime)) {
          trip += tripSize;
        }
        if(trip[stopOffset].departure_ < label.arrivalTime) continue;

        RouteLabel newLabel;
        newLabel.trip = trip;
        newLabel.parentIndex = j;
        newRouteBag.merge(newLabel);
      }
      stopOffset++;
      stop = query.tt_.route_stops_[stopOffset];
      for(RouteLabel& label : newRouteBag.labels) {
        Label newLabel;
        newLabel.arrivalTime = label.trip[stopOffset].arrival_;
        newLabel.parentStation = query.tt_.route_stops_[label.parentStop];
        newLabel.parentIndex = label.parentIndex;
        newLabel.parentDepartureTime = label.trip[label.parentStop].departure_;
        newLabel.routeId = routeId;
        arrival_by_route(stop, newLabel);

      }
    }

  }
}

Bag* McRaptor::currentRound() {
  return result[round];
}

Bag* McRaptor::previousRound() {
  return result[round - 1];
}

void McRaptor::startNewRound() {
  round += 1;
}

// searches through all routes in station
//
// returns vector of pairs with route itself and index to the given station
inline std::vector<std::pair<route_id, route_stops_index>> McRaptor::getRoutesTimesForStop(stop_id stop) {
  std::vector<std::pair<route_id, route_stops_index>> result = std::vector<std::pair<route_id, route_stops_index>>();
  // go through all routes for the given station using the first route as base
  // and adding offset to this base until the base + offset = count of routes in current station
  for(route_id routeId = query.tt_.stop_routes_[stop]; routeId < query.tt_.stops_[stop].route_count_; routeId++) {
    // extract this route form timetable using its id
    raptor_route route = query.tt_.routes_[routeId];
    // go through stops of this route
    for (route_stops_index r_stop_offset = route.index_to_route_stops_; r_stop_offset < route.stop_count_;
         r_stop_offset++) {
      // add the station and the founded station to the result
      if(query.tt_.route_stops_[r_stop_offset] == stop) {
        result.emplace_back(std::make_pair(routeId, r_stop_offset));
      }
    }
  }
  return result;
}

void McRaptor::invoke_cpu_raptor() {
  //auto const& tt = McRaptor::query.tt_;

  /*Rounds& result = query.result();
  cpu_mark_store* stopsForTransfers = new cpu_mark_store(tt.stop_count());
  stopsUpdatedByTransfers = stopsForTransfers;
  cpu_mark_store* stopsForRoutes = new cpu_mark_store(tt.route_count());
  stopsUpdatedByRoute = stopsForRoutes;*/

  //cpu_mark_store route_marks(tt.route_count());

  init_arrivals();
  relax_transfers();

  for (auto i = 0; i < max_raptor_round; i++) {
    startNewRound();
    collect_routes_serving_updated_stops();
    scan_routes();
    if(stopsForRoutes.empty()) {
      break;
    }
    startNewRound();
    relax_transfers();
  }

  /*for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
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
  }*/
}

/*trip_count get_earliest_trip(raptor_timetable const& tt,
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

*//*void McRaptor::init_arrivals() {

    // Don't set the values for the earliest arrival, as the footpath update
    // in the first round will use the values in conjunction with the
    // footpath lengths without transfertime leading to invalid results.
    // Not setting the earliest arrival values should (I hope) be correct.

    startNewRound();
Label newLabel(0, query.source_time_begin_, 0);
arrival_by_route(query.source_, newLabel);
startNewRound();

//result[0][q.source_].merge(newLabel);
//station_marks.mark(q.source_);

//  for (auto const& add_start : q.add_starts_) {
//    time const add_start_time = q.source_time_begin_ + add_start.offset_;
//    Label addStartLabel(0, add_start_time, 0);
//    result[0][add_start.s_id_].merge(addStartLabel);
//    station_marks.mark(add_start.s_id_);
//  }
}*//*


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

    *//*
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
     *//*

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
}*/

}  // namespace motis::mcraptor

