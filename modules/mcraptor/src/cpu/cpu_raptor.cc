#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/bag.h"
#include <map>

namespace motis::mcraptor {

template <class T, class L>
void mc_raptor<T, L>::init_arrivals() {
  static_cast<T*>(this)->init_arrivals();
}

template <class T, class L>
void mc_raptor<T, L>::arrival_by_route(stop_id stop, L& new_label) {
  if(new_label.arrival_time_ < source_time_begin_) {
    return;
  }
  // ??? checking for empty
  // check if this label may be dominated by labels on the last stations
  const auto& t = static_cast<T*>(this)->targets_;
  for(stop_id target : t) {
    if(transfer_labels_[target].dominates(new_label)) {
      return;
    }
  }
  if(!route_labels_[stop].merge(new_label)) {
    return;
  }
  // add indominated label to the bags
  transfer_labels_[stop].merge(new_label);
  current_round()[stop].merge_undominated(new_label);
  // mark the station
  stops_for_routes_.mark(stop);
}

template <class T, class L>
void mc_raptor<T, L>::arrival_by_transfer(stop_id stop, L& new_label) {
  if(new_label.arrival_time_ < source_time_begin_) {
    return;
  }
  // checking for empty??
  // check if this label may be dominated by other existing labels
  const auto& t = static_cast<T*>(this)->targets_;
  for(stop_id target : t) {
    if(transfer_labels_[target].dominates(new_label)) {
      return;
    }
  }
  if(!transfer_labels_[stop].merge(new_label)) {
    return;
  }
  // add indominated label to the bag
  current_round()[stop].merge_undominated(new_label);
  // mark current station
  stops_for_transfers_.mark(stop);
}

template <class T, class L>
void mc_raptor<T, L>::relax_transfers() {
  stops_for_transfers_.reset();
  std::fill(routes_serving_updated_stops_.begin(), routes_serving_updated_stops_.end(), invalid<route_stops_index>);
  // iterate through all station and find marked
  for(stop_id stop = 0; stop < query_.tt_.stop_count(); ++stop) {
    if(!stops_for_routes_.marked(stop)) {
      continue;
    }
    stops_for_transfers_.mark(stop);
    bag<L>& bag = previous_round()[stop];
    current_round()[stop].labels_.resize(bag.size());
    for(size_t i = 0; i < bag.size(); ++i) {
      current_round()[stop][i] = L(bag[i], stop, i);
    }
  }

  for(stop_id stop = 0; stop < query_.tt_.stop_count(); ++stop) {
    if (!stops_for_routes_.marked(stop)) {
      continue;
    }
    bag<L>& bag = previous_round()[stop];

    // iterate through footpaths - coming from the update_footpath()
    auto index_into_transfers = query_.tt_.stops_[stop].index_to_transfers_;
    auto next_index_into_transfers = query_.tt_.stops_[stop + 1].index_to_transfers_;
    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {
      auto const& to_stop = query_.tt_.footpaths_[current_index].to_;
      auto const& duration = query_.tt_.footpaths_[current_index].duration_;
      static_cast<T*>(this)->init_new_label(bag, stop, duration, to_stop);
    }
  }
}

template <class T, class L>
void mc_raptor<T, L>::collect_routes_serving_updated_stops() {
  // find marked stations
  for(stop_id stop = 0; stop < query_.tt_.stop_count(); stop++) {
    if (!stops_for_transfers_.marked(stop)) {
      continue;
    }
    // find the offset in form of stop_id where the route stops by current station (getRoutesTimeForStops method)
    for(std::pair<route_id , route_stops_index> s : get_routes_times_for_stop(stop)) {
      // id of this route
      const route_id route_id = s.first;
      // route object itself
      const raptor_route route = query_.tt_.routes_[route_id];
      // this offset where the route stops on this station
      const route_stops_index stop_offset = s.second;
      // if it is the last station
      if(stop_offset == route.stop_count_ - 1) {
        continue;
      }
      // write to this route the earliest stop from both
      routes_serving_updated_stops_[route_id] = std::min(routes_serving_updated_stops_[route_id], stop_offset);
    }
  }
}

template <class T, class L>
void mc_raptor<T, L>::scan_routes() {
  stops_for_routes_.reset();
  for(auto i = 0; i < routes_serving_updated_stops_.size(); ++i) {
    route_id route_id = i;
    route_stops_index stop_offset = routes_serving_updated_stops_[i];
    if(!valid(stop_offset)) {
      continue;
    }
    raptor_route route = query_.tt_.routes_[route_id];
    stop_id stop = query_.tt_.route_stops_[route.index_to_route_stops_ + stop_offset];
    const stop_count trip_size = route.stop_count_;

    // event represents the arrival and departure times of route in its single station
    const stop_time* first_trip = &query_.tt_.stop_times_[route.index_to_stop_times_];
    const stop_time* last_trip = &query_.tt_.stop_times_[route.index_to_stop_times_ + trip_size * (route.trip_count_ - 1)];

    static_cast<T*>(this)->scan_route(stop, stop_offset, trip_size, first_trip, last_trip, route, route_id);

  }
}

template <class T, class L>
bag<L>* mc_raptor<T, L>::current_round() {
  return result_[round_];
}

template <class T, class L>
bag<L>* mc_raptor<T, L>::previous_round() {
  return result_[round_ - 1];
}

template <class T, class L>
void mc_raptor<T, L>::start_new_round() {
  round_ += 1;


  //test output

  /*std::cout << std::endl << std::endl << std::endl << "Round: " << round_ - 2 << std::endl;

  int r_k = round_ - 2;
  if(r_k >= 0) {
    bag<L>& target_bag = result_[r_k][query_.target_];
    if (target_bag.is_valid()) {
      stop_id current_station = query_.target_;
      size_t label = 0;
      while (valid(current_station) && r_k >= 0 && result_[r_k][current_station].is_valid()) {
        std::cout << current_station << "("
                  << result_[r_k][current_station][label].arrival_time_ << ") <- ";
        stop_id parent_station =
            result_[r_k][current_station][label].parent_station_;
        label = result_[r_k][current_station][label].parent_label_index_;
        current_station = parent_station;
        r_k--;
      }
    }
  }*/
}

// searches through all routes in station
//
// returns vector of pairs with route itself and index to the given station
template <class T, class L>
inline std::vector<std::pair<route_id, route_stops_index>>& mc_raptor<T, L>::get_routes_times_for_stop(stop_id stop_id) {
  routes_times_for_stop_.clear();
  // go through all routes for the given station using the first route as base
  // and adding offset to this base until the base + offset = count of routes in current station
  raptor_stop stop = query_.tt_.stops_[stop_id];
  for(stop_routes_index stop_route_id = stop.index_to_stop_routes_; stop_route_id < stop.index_to_stop_routes_ + stop.route_count_; ++stop_route_id) {
    // extract this route form timetable using its id
    route_id route_id = query_.tt_.stop_routes_[stop_route_id];
    raptor_route route = query_.tt_.routes_[route_id];
    // go through stops of this route
    for (route_stops_index stop_offset = 0; stop_offset < route.stop_count_;
         stop_offset++) {
      // add the station and the founded station to the result
      if(query_.tt_.route_stops_[stop_offset + route.index_to_route_stops_] == stop_id) {
        routes_times_for_stop_.emplace_back(std::make_pair(route_id, stop_offset));
      }
    }
  }
  return routes_times_for_stop_;
}

template <class T, class L>
void mc_raptor<T, L>::set_query_source_time(time other_time) {
  source_time_begin_ = other_time;
}

template <class T, class L>
void mc_raptor<T, L>::reset() {
  round_ = -1;
  std::fill(routes_serving_updated_stops_.begin(), routes_serving_updated_stops_.end(), invalid<route_stops_index>);
  stops_for_routes_.reset();
  stops_for_transfers_.reset();
}

template <class T, class L>
void mc_raptor<T, L>::invoke_cpu_raptor() {
  /*std::cout << "Target: " << query_.target_ << std::endl;
  std::cout << "Source: " << query_.source_ << std::endl;*/

  init_arrivals();
  relax_transfers();

  for (auto i = 0; i < max_raptor_round * 2; ++i) {
    start_new_round();
    collect_routes_serving_updated_stops();
    scan_routes();
    // TODO: check if there are no marks true
    if(stops_for_routes_.no_marked_stops()) {
      break;
    }
    start_new_round();
    relax_transfers();
  }

  static_cast<T*>(this)->init_parents();

}


//departure mc_raptor

void mc_raptor_departure::init_arrivals() {
  start_new_round();

  if (query_.source_ == 0) {
    for (raptor_edge edge : query_.raptor_edges_start_) {
      // std::cout << "EDGE from: " << edge.from_ << "; to: " << edge.to_ << "; time: " << edge.time_ << std::endl;
      time edge_to_time = source_time_begin_ + edge.duration_;
      label_departure new_label(invalid<time>, edge_to_time, round_);
      new_label.parent_station_ = edge.to_;
      new_label.journey_departure_time_ = edge_to_time;
      arrival_by_route(edge.to_, new_label);
    }
  } else {
    label_departure new_label(invalid<time>, source_time_begin_, round_);
    new_label.parent_station_ = query_.source_;
    new_label.journey_departure_time_ = source_time_begin_;
    arrival_by_route(query_.source_, new_label);

    for (auto const& add_start : query_.add_starts_) {
      time add_start_time = source_time_begin_ + add_start.offset_;
      new_label = label_departure(invalid<time>, add_start_time, round_);
      new_label.parent_station_ = add_start.s_id_;
      new_label.journey_departure_time_ = add_start_time;
      // std::cout << "Add start: " << add_start.s_id_ << std::endl;
      arrival_by_route(add_start.s_id_, new_label);
    }
  }
  start_new_round();
}

void mc_raptor_departure::init_new_label(bag<label_departure> bag,
                                         stop_id stop, time8 duration, stop_id to_stop) {
  for(size_t i = 0; i < bag.size(); ++i) {
    label_departure new_label;
    new_label.arrival_time_ = bag[i].arrival_time_ + duration;
    new_label.parent_station_ = stop;
    new_label.parent_label_index_ = i;
    new_label.changes_count_ = round_;
    new_label.footpath_duration_ = duration;
    new_label.journey_departure_time_ = bag[i].journey_departure_time_;
    arrival_by_transfer(to_stop, new_label);
  }
}

void mc_raptor_departure::scan_route(stop_id stop, route_stops_index stop_offset,
                                     const stop_count trip_size, const stop_time* first_trip,
                                     const stop_time* last_trip, raptor_route route,
                                     route_id route_id) {
  route_bag new_route_bag;

  while(stop_offset < trip_size - 1) {
    for (size_t j = 0; j < previous_round()[stop].size(); ++j) {
      const label_departure& label = previous_round()[stop][j];
      const stop_time* trip = first_trip;
      trip_id current_trip_id = 0;
      while ((trip < last_trip) && (trip[stop_offset].departure_ < label.arrival_time_)) {
        trip += trip_size;
        current_trip_id++;
      }

      time trip_departure = trip[stop_offset].departure_;
      if (!valid(trip_departure) || trip_departure < label.arrival_time_) {
        continue;
      }

      route_label new_label;
      new_label.trip_ = trip;
      new_label.parent_label_index_ = j;
      new_label.parent_stop_ = stop;
      new_label.current_trip_id_ = current_trip_id;  // = tripId;
      new_route_bag.merge(new_label);
    }
    stop_offset++;
    stop = query_.tt_.route_stops_[route.index_to_route_stops_ + stop_offset];
    for (route_label& r_label : new_route_bag.labels_) {
      label_departure new_label;
      new_label.arrival_time_ = r_label.trip_[stop_offset].arrival_;
      new_label.parent_station_ = r_label.parent_stop_;
      new_label.parent_label_index_ = r_label.parent_label_index_;
      new_label.route_id_ = route_id;
      new_label.stop_offset_ = stop_offset;
      new_label.current_trip_id_ = r_label.current_trip_id_;
      new_label.changes_count_ = round_;
      new_label.journey_departure_time_ = previous_round()[r_label.parent_stop_][r_label.parent_label_index_].journey_departure_time_;
      arrival_by_route(stop, new_label);
    }
  }
}

void mc_raptor_departure::init_parents(){

}

//arrival mc_raptor

void mc_raptor_arrival::init_arrivals() {
  start_new_round();

  if (query_.target_ == 1) {
    for (raptor_edge edge : query_.raptor_edges_end_) {
      time edge_to_time = source_time_begin_ - edge.duration_;
      label_arrival new_label(invalid<time>, edge_to_time, round_);
      new_label.backward_parent_station = edge.to_;
      arrival_by_route(edge.to_, new_label);
    }
  } else {
    label_arrival new_label(invalid<time>, query_.source_time_end_, round_);
    new_label.backward_parent_station = query_.target_;
    arrival_by_route(query_.target_, new_label);
  }
  start_new_round();
}

void mc_raptor_arrival::init_new_label(bag<label_arrival> bag,
                                       stop_id stop, time8 duration, stop_id to_stop) {
  for(size_t i = 0; i < bag.size(); ++i) {
    label_arrival new_label;
    new_label.departure_time_ = bag[i].departure_time_ - duration;
    new_label.backward_parent_station = stop;
    new_label.backward_parent_label_index_ = i;
    new_label.footpath_duration_ = duration;
    arrival_by_transfer(to_stop, new_label);
  }
}

void mc_raptor_arrival::scan_route(stop_id stop, route_stops_index stop_offset,
                const stop_count trip_size, const stop_time* first_trip,
                const stop_time* last_trip, raptor_route route,
                route_id route_id) {
  route_bag new_route_bag;

  while(stop_offset > 0) {
    for(size_t j = 0; j < previous_round()[stop].size(); ++j) {
      const label_arrival& label = previous_round()[stop][j];
      const stop_time* trip = last_trip;
      trip_id current_trip_id = route.trip_count_;
      while((trip > first_trip) && (trip[stop_offset].arrival_ > label.departure_time_)) {
        trip -= trip_size;
        current_trip_id--;
      }

      time trip_arrival = trip[stop_offset].arrival_;
      if(!valid(trip_arrival) || trip_arrival < label.departure_time_) {
        continue;
      }

      route_label new_label;
      new_label.trip_ = trip;
      new_label.parent_label_index_ = j;
      new_label.parent_stop_ = stop;
      new_label.current_trip_id_ = current_trip_id ; // = tripId;
      new_route_bag.merge(new_label);
    }
    stop_offset--;
    stop = query_.tt_.route_stops_[route.index_to_route_stops_ + stop_offset];
    for(route_label& r_label : new_route_bag.labels_) {
      label_arrival new_label;
      new_label.departure_time_ = r_label.trip_[stop_offset].departure_;
      new_label.backward_parent_station = r_label.parent_stop_;
      new_label.backward_parent_label_index_ = r_label.parent_label_index_;
      new_label.route_id_ = route_id;
      new_label.stop_offset_ = stop_offset;
      new_label.current_trip_id_ = r_label.current_trip_id_;
      arrival_by_route(stop, new_label);
    }
  }
}
void mc_raptor_arrival::init_parents(){
  std::cout << "Source: " << query_.source_ << std::endl;
  for(int r_m = 0; r_m <= round_; r_m++) {
    std::cout << "Round " << r_m << "; Size " << result_[r_m][query_.source_].labels_.size() << std::endl;
    if(r_m % 2 == 1) { //TODO fix this
      continue;
    }
    std::cout << "Starting parents init" << std::endl;
    bag<label_arrival> start_bag = result_[r_m][query_.source_];
    for (int l_id = 0; l_id < start_bag.labels_.size(); l_id++) {
      label_arrival* prev_label = &start_bag.labels_[l_id];
      int prev_label_id = query_.source_;
      int prev_parent_label_index = l_id;
      prev_label->changes_count_ = 0;
      prev_label->parent_station_ = query_.source_;

      for (int i = 1; i < r_m; i++) {
        int r = r_m - i;
        std::cout << "Parents for round " << r << std::endl;
        std::cout << "BS: " << prev_label->backward_parent_station << "; BSI: " << prev_label->backward_parent_label_index_ << "; Size: " << result_[r][prev_label->backward_parent_station].size() << std::endl;
        label_arrival* current_label = &result_[r][prev_label->backward_parent_station][prev_label->backward_parent_label_index_];
        current_label->parent_station_ = prev_label_id;
        current_label->parent_label_index_ = prev_parent_label_index;
        current_label->changes_count_ = i;
        prev_label_id = prev_label->backward_parent_station;
        prev_parent_label_index = prev_label->backward_parent_label_index_;
        prev_label = current_label;
      }
    }
  }

  //TODO reverse result
}

}  // namespace motis::mcraptor

