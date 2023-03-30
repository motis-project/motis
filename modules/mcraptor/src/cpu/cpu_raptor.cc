#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/bag.h"
#include <map>

namespace motis::mcraptor {


template <class T, class L>
inline bool mc_raptor<T, L>::is_label_pruned(stop_id stop, L& new_label) {

  if(!new_label.is_valid()) {
    return true;
  }

  const std::vector<stop_id>& t = static_cast<T*>(this)->targets_;
  bool merged_target = false;
  for (stop_id target : t) {
    if (stop == target) {
      if (!target_labels_[stop].merge(new_label)) {
        return true;
      }
      else {
        merged_target = true;
      }
    } else if (target_labels_[target].dominates(new_label)) {
      return true;
    }
  }
  if (!merged_target && !target_labels_[stop].merge(new_label)) {
    return true;
  }

  return false;
}

template <class T, class L>
void mc_raptor<T, L>::arrival_by_route(stop_id stop, L& new_label, bool from_equal_station) {
  if (query_.source_ == 0) {
    bool found = false;
    time duration = 0;
    for (raptor_edge target_edge: query_.raptor_edges_end_) {
      if (target_edge.from_ == stop) {
        found = true;
        duration = target_edge.duration_;
        break;
      }
    }
    if (found) {
      new_label.current_target_ = stop;
      new_label.arrival_time_ += duration;
    }
  }

  if(is_label_pruned(stop, new_label)) {
    return;
  }

  // add indominated label to the bag
  current_round()[stop].merge_undominated(new_label);
  // mark the station
  stops_for_routes_.mark(stop);

  // Check equal stations if there is target among them
  if(query_.forward_ && !from_equal_station && query_.source_ != 0) {
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
      init_new_label(bag, stop, duration, to_stop);
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
      bool is_circled_route = false;
      // id of this route
      const route_id route_id = s.route_id;
      // route object itself
      const raptor_route route = query_.tt_.routes_[route_id];
      // this offset where the route stops on this station
      const route_stops_index stop_offset = s.stop_offset;
      // check if this a circled route ==> means s1 -> s2 -> s3 -> s2 -> s1
      const stop_count trip_size = route.stop_count_;
      if (query_.tt_.route_stops_[route.index_to_route_stops_ + (trip_size / 2) - 1] == query_.tt_.route_stops_[route.index_to_route_stops_ + (trip_size / 2) + 1]) {
        is_circled_route = true;
      }
      // write to this route the earliest stop from both
      if (is_circled_route) {
        auto const actual_stop_offset = stop_offset - trip_size / 2;
        auto const actual_written_stop_offset = routes_serving_updated_stops_[route_id];
        if (stop_offset > trip_size / 2) {
          // after && after
          if (actual_written_stop_offset > trip_size / 2) {
            routes_serving_updated_stops_[route_id] =
                std::min(routes_serving_updated_stops_[route_id], stop_offset);
            // after && before
          } else {
            if(actual_stop_offset < routes_serving_updated_stops_[route_id]) {
              routes_serving_updated_stops_[route_id] = stop_offset;
            }
          }
        } else if (stop_offset == trip_size / 2) {
          routes_serving_updated_stops_[route_id] =
              std::min(routes_serving_updated_stops_[route_id], stop_offset);
        } else {
          // before && before
          if (actual_written_stop_offset < trip_size / 2) {
            routes_serving_updated_stops_[route_id] =
                std::min(routes_serving_updated_stops_[route_id], stop_offset);
            // before && after
          } else {
            if (stop_offset < actual_written_stop_offset) {
              routes_serving_updated_stops_[route_id] = stop_offset;
            }
          }
        }
      } else {
          routes_serving_updated_stops_[route_id] =
              get_earliest(routes_serving_updated_stops_[route_id], stop_offset);
      }
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

    scan_route(stop, stop_offset, trip_size, first_trip, last_trip, route, route_id);

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
void mc_raptor<T, L>::set_current_start_edge(motis::mcraptor::raptor_edge edge) {
  current_source_edge = edge;
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

}


//departure mc_raptor

void mc_raptor_departure::init_arrivals() {
  start_new_round();

  label_departure new_label;
  new_label = label_departure(invalid<time>, source_time_begin_, round_);
  new_label.parent_station_ = current_source_edge.to_;
  new_label.journey_departure_time_ = source_time_begin_ - current_source_edge.duration_;
  //std::cout << "EDGE from: " << query_.meta_info_.raptor_id_to_eva_.at(current_source_edge.from_) << "; to: " << query_.meta_info_.raptor_id_to_eva_.at(current_source_edge.to_) << "; time: " << current_source_edge.duration_ << "; arrival time: " << source_time_begin_ <<  std::endl;
  arrival_by_route(current_source_edge.to_, new_label);

  if (query_.source_ != 0) {
    for (auto const& add_start : query_.add_starts_) {
      if (add_start.s_id_ == query_.source_) {
        continue;
      }
      new_label = label_departure(invalid<time>, source_time_begin_, round_);
      new_label.parent_station_ = add_start.s_id_;
      new_label.journey_departure_time_ = source_time_begin_;
      //std::cout << "EDGE from: " << query_.meta_info_.raptor_id_to_eva_.at(add_start.s_id_) << "; to: " << query_.meta_info_.raptor_id_to_eva_.at(add_start.s_id_) << "; arrival time: " << source_time_begin_ <<  std::endl;
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

      //bool from_source = std::find(query_.meta_info_.equivalent_stations_[query_.source_].begin(), query_.meta_info_.equivalent_stations_[query_.source_].end(), label.parent_station_) != query_.meta_info_.equivalent_stations_[query_.source_].end() &&
                         std::find(query_.meta_info_.equivalent_stations_[query_.source_].begin(), query_.meta_info_.equivalent_stations_[query_.source_].end(), stop) != query_.meta_info_.equivalent_stations_[query_.source_].end();

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

route_stops_index mc_raptor_departure::get_earliest(route_stops_index a, route_stops_index b) {
  return std::min(a, b);
}

//arrival mc_raptor

void mc_raptor_backward::init_arrivals() {
  start_new_round();
  //TODO MERGE WITH INTERMODAL

   label_backward new_label;

   // TODO FIX THIS
   auto target_add_starts = get_add_starts(query_.meta_info_, query_.target_, query_.use_start_footpaths_, query_.use_dest_metas_);
   for (auto const& add_start : target_add_starts) {
     new_label = label_backward(source_time_begin_, source_time_begin_, round_);
     new_label.backward_parent_station_ = add_start.s_id_;
     arrival_by_route(add_start.s_id_, new_label);
   }

  start_new_round();
}

void mc_raptor_backward::init_new_label(bag<label_backward> bag,
                                         stop_id stop, time8 duration, stop_id to_stop) {
  for(auto& l : bag.labels_) {
    label_backward new_label;
    new_label.departure_time_ = l.departure_time_ - duration;
    new_label.parent_arrival_time_ = l.departure_time_;
    new_label.backward_parent_station_ = stop;
    new_label.changes_count_ = round_;
    new_label.footpath_duration_ = duration;
    new_label.journey_arrival_time_ = l.journey_arrival_time_;
    arrival_by_transfer(to_stop, new_label);
  }
}

void mc_raptor_backward::scan_route(stop_id stop, route_stops_index stop_offset,
                                     const stop_count trip_size, const stop_time* first_trip,
                                     const stop_time* last_trip, raptor_route route,
                                     route_id route_id) {
  bag<route_label> new_route_bag;

  while(stop_offset > 0) {
    for (auto& label : previous_round()[stop].labels_) {
      const stop_time* trip = last_trip;
      trip_id current_trip_id = (last_trip - first_trip) / trip_size;
      while ((trip > first_trip) && (label.departure_time_ < trip[stop_offset].arrival_)) {
        trip -= trip_size;
        current_trip_id--;
      }

      time trip_arrival = trip[stop_offset].arrival_;
      if (!valid(trip_arrival) || label.departure_time_ < trip_arrival) {
        continue;
      }

      route_label new_label;
      new_label.forward = false;
      new_label.trip_ = trip;
      new_label.parent_journey_arrival_time_ = label.journey_arrival_time_;
      new_label.parent_stop_ = stop;
      new_label.parent_stop_offset_ = stop_offset;
      new_label.parent_arrival_time_ = trip_arrival;
      new_label.current_trip_id_ = current_trip_id;  // = tripId;
      new_route_bag.merge(new_label);
    }
    stop_offset--;
    stop = query_.tt_.route_stops_[route.index_to_route_stops_ + stop_offset];
    for (auto& r_label : new_route_bag.labels_) {
      label_backward new_label;
      new_label.departure_time_ = r_label.trip_[stop_offset].departure_;
      new_label.backward_parent_station_ = r_label.parent_stop_;
      new_label.route_id_ = route_id;
      new_label.stop_offset_ = stop_offset;
      new_label.current_trip_id_ = r_label.current_trip_id_;
      new_label.changes_count_ = round_;
      new_label.journey_arrival_time_ = r_label.parent_journey_arrival_time_;
      new_label.parent_stop_offset_ = r_label.parent_stop_offset_;
      new_label.parent_arrival_time_ = r_label.parent_arrival_time_;
      arrival_by_route(stop, new_label);
    }
  }
}

void mc_raptor_backward::init_parents() {
//  std::cout << "Source: " << query_.source_ << std::endl;
//  std::cout << "Target: " << query_.target_ << std::endl;


  stop_id source = query_.source_;
  rounds<label_backward> new_res(stop_count_);
  label_backward invalid_label;
  for(int r_m = 0; r_m <= max_raptor_round * 2; r_m++) {
    if(r_m % 2 == 0) { //TODO fix this
      continue;
    }

//    if(result_[r_m][source].labels_.size() > 0) {
//      std::cout << "Round " << r_m << "; Size " << result_[r_m][source].labels_.size() << std::endl;
//    }
//    if(r_m % 2 == 1) { //TODO fix this
//      continue;
//    }
//    std::cout << "Starting parents init" << std::endl;
    bag<label_backward> start_bag = result_[r_m][source];
    for (auto& start_label : start_bag.labels_) {
      int parent_station = source;
      int current_station = source;
      int changes = start_label.changes_count_;
      int journey_departure = start_label.departure_time_;
      label_backward current_label = start_label;
      label_backward parent_label = start_label;

//      if(r_m % 2 == 0) {
//        label_backward first_transfer_label(current_label, source);
//        parent_label = first_transfer_label;
//        current_label = first_transfer_label;
//        changes += 1;
//      }

//      std::cout << "Path" << "(" << journey_departure << "; " << current_label.journey_arrival_time_ << ")" <<  ": ";

      int forward_changes_count = 0;
      while (forward_changes_count < changes) {
        label_backward parent_label_backup = current_label;
        if(forward_changes_count != 0) {
          current_label.current_trip_id_ = parent_label.current_trip_id_;
          current_label.stop_offset_ = parent_label.parent_stop_offset_;
          current_label.arrival_time_ = parent_label.parent_arrival_time_;
          current_label.route_id_ = parent_label.route_id_;
          current_label.footpath_duration_ = parent_label.footpath_duration_;
        }
        else {
          current_label.arrival_time_ = journey_departure;
        }
        parent_label = parent_label_backup;

        current_label.changes_count_ = forward_changes_count;
        current_label.parent_station_ = parent_station;
        current_label.journey_departure_time_ = journey_departure;
//        current_label.arrival_time_ = // ??????????????
//        new_res[forward_changes_count]->labels_.push_back(current_label);
//        std::cout << current_station << "(" << parent_label.departure_time_ << "; " << current_label.arrival_time_ << ")" << " -> ";

        if(changes - forward_changes_count == 1) {
          current_label.journey_arrival_time_ = current_label.arrival_time_;
        }

        new_res[forward_changes_count][current_station].labels_.push_back(current_label);
        forward_changes_count++;
        parent_station = current_station;
        current_station = current_label.backward_parent_station_;
        current_label = result_[changes - forward_changes_count][current_station].get_fastest_backward_label(current_label.arrival_time_, invalid_label);
        if (!valid(current_label.backward_parent_station_)) {
          break;
        }
      }
//      std::cout << std::endl;
    }
  }

  result_.change(new_res);
}

route_stops_index mc_raptor_backward::get_earliest(route_stops_index a, route_stops_index b) {
  if(!valid(a)) {
    return b;
  }

  if(!valid(b)) {
    return a;
  }

  return std::max(a, b);
}


}  // namespace motis::mcraptor

