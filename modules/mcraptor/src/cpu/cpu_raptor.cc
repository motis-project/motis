#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/Bag.h"
#include <map>

namespace motis::mcraptor {

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
  // add indominated label to the bag
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
    // TODO: check
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

    // TODO: check
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
}

}  // namespace motis::mcraptor

