#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/bag.h"
#include <map>

namespace motis::mcraptor {

template <class T, class L>
void mc_raptor<T, L>::init_arrivals() {
  static_cast<T*>(this)->init_arrivals();
}
template <class T, class L>
inline bool mc_raptor<T, L>::is_label_pruned(stop_id stop, L& new_label) {
  if(new_label.arrival_time_ < source_time_begin_) {
    return true;
  }

  const std::vector<stop_id>& t = static_cast<T*>(this)->targets_;
  bool merged_target = false;
  for(stop_id target : t) {
    if(stop == target) {
      if(!target_labels_[stop].merge(new_label)) {
        return true;
      }
      else {
        merged_target = true;
      }
    }
    else if(target_labels_[target].dominates(new_label)) {
      return true;
    }
  }
  if(!merged_target && !target_labels_[stop].merge(new_label)) {
    return true;
  }

  return false;
}

template <class T, class L>
void mc_raptor<T, L>::arrival_by_route(stop_id stop, L& new_label, bool from_equal_station) {
  if(is_label_pruned(stop, new_label)) {
    return;
  }

  // add indominated label to the bag
  current_round()[stop].merge_undominated(new_label);
  // mark the station
  stops_for_routes_.mark(stop);

  // Check equal stations if there is target among them
  if(!from_equal_station) {
    const std::vector<stop_id>& t = static_cast<T*>(this)->targets_;
    for (stop_id s : query_.meta_info_.equivalent_stations_[stop]) {
      if (s == stop) {
        continue;
      }
      auto sf = std::find(t.begin(), t.end(), s);
      if (sf != t.end()) {
        arrival_by_route(s, new_label, true);
      }
    }
  }

}

template <class T, class L>
void mc_raptor<T, L>::arrival_by_transfer(stop_id stop, L& new_label) {
  if(is_label_pruned(stop, new_label)) {
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
  for(stop_id stop = 0; stop < stop_count_; ++stop) {
    if(!stops_for_routes_.marked(stop)) {
      continue;
    }
    stops_for_transfers_.mark(stop);
    bag<L>& prev_bag = previous_round()[stop];
    bag<L>& current_bag = current_round()[stop];
    size_t prev_size = prev_bag.labels_.size();
    current_bag.labels_.resize(prev_size);
    for(size_t i = 0; i < prev_size; ++i) {
      current_bag.labels_[i] = L(prev_bag.labels_[i], stop);
    }
  }

  for(stop_id stop = 0; stop < stop_count_; ++stop) {
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
  for(stop_id stop = 0; stop < stop_count_; stop++) {
    if (!stops_for_transfers_.marked(stop)) {
      continue;
    }
    // find the offset in form of stop_id where the route stops by current station (getRoutesTimeForStops method)
    for(route_with_stop_offset s : query_.meta_info_.routes_times_for_stop[stop]) {
      // id of this route
      const route_id route_id = s.route_id;
      // route object itself
      const raptor_route route = query_.tt_.routes_[route_id];
      // this offset where the route stops on this station
      const route_stops_index stop_offset = s.stop_offset;
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
    label_departure new_label;

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
  for(auto& l : bag.labels_) {
    label_departure new_label;
    new_label.arrival_time_ = l.arrival_time_ + duration;
    new_label.parent_station_ = stop;
    new_label.changes_count_ = round_;
    new_label.footpath_duration_ = duration;
    new_label.journey_departure_time_ = l.journey_departure_time_;
    arrival_by_transfer(to_stop, new_label);
  }
}

void mc_raptor_departure::scan_route(stop_id stop, route_stops_index stop_offset,
                                     const stop_count trip_size, const stop_time* first_trip,
                                     const stop_time* last_trip, raptor_route route,
                                     route_id route_id) {
  bag<route_label> new_route_bag;

  while(stop_offset < trip_size - 1) {
    for (auto& label : previous_round()[stop].labels_) {
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
      new_label.parent_journey_departure_time_ = label.journey_departure_time_;
      new_label.parent_stop_ = stop;
      new_label.current_trip_id_ = current_trip_id;  // = tripId;
      new_route_bag.merge(new_label);
    }
    stop_offset++;
    stop = query_.tt_.route_stops_[route.index_to_route_stops_ + stop_offset];
    for (auto& r_label : new_route_bag.labels_) {
      label_departure new_label;
      new_label.arrival_time_ = r_label.trip_[stop_offset].arrival_;
      new_label.parent_station_ = r_label.parent_stop_;
      new_label.route_id_ = route_id;
      new_label.stop_offset_ = stop_offset;
      new_label.current_trip_id_ = r_label.current_trip_id_;
      new_label.changes_count_ = round_;
      new_label.journey_departure_time_ = r_label.parent_journey_departure_time_;
      arrival_by_route(stop, new_label);
    }
  }
}

void mc_raptor_departure::init_parents(){

}

//arrival mc_raptor

}  // namespace motis::mcraptor

