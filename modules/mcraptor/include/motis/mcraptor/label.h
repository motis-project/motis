#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {

template <class T>
struct label {

  static const time MAX_DIFF_FOR_LESS_TRANSFERS = 120;

  inline int arrival_time_rule(label& other) {
    return static_cast<T*>(this)->arrival_time_rule(other);
  }

  inline int journey_departure_time_rule(label& other) {
    return static_cast<T*>(this)->journey_departure_time_rule(other);
  }

  inline int changes_count_rule(label& other) {
    return static_cast<T*>(this)->changes_count_rule(other);
  }

  inline int travel_duration_rule(label& other) {
    return static_cast<T*>(this)->travel_duration_rule(other);
  }

  // Parameters
  time journey_departure_time_ = invalid<time>;
  size_t changes_count_ = invalid<size_t>;
  time arrival_time_ = invalid<time>;

  // Parent info
  stop_id parent_station_ = invalid<stop_id>;

  // Current trip info
  route_id route_id_ = invalid<route_id>;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;
  time footpath_duration_ = invalid<time>;

  label() {
  }

  label(time journey_departure_time, time arrival_time, size_t changes_count) : journey_departure_time_(journey_departure_time),
                                                                     arrival_time_(arrival_time),
                                                                     changes_count_(changes_count) { }

  // to create labels for current round from labels from previous round for certain station
  label(label& parent_label, stop_id parent_station) : arrival_time_(parent_label.arrival_time_),
                                                                            changes_count_(parent_label.changes_count_ + 1),
                                                                            journey_departure_time_(parent_label.journey_departure_time_),
                                                                            parent_station_(parent_station){ }

  bool dominates(label& other) {
    int domination_arrival_time = arrival_time_rule(other);
    int domination_journey_departure_time = journey_departure_time_rule(other);
    int domination_changes_count = changes_count_rule(other);
    int domination_travel_duration = travel_duration_rule(other);
//    if(domination_arrival_time == 0 && domination_journey_departure_time == 0 && domination_changes_count == 0
//        && parent_station_ == other.parent_station_ && route_id_ == other.route_id_ && current_trip_id_ == other.current_trip_id_
//        && stop_offset_ == other.stop_offset_ && footpath_duration_ == other.footpath_duration_) {
//      return true;
//    }

    return domination_arrival_time >= 0 && domination_changes_count >= 0 && domination_journey_departure_time >= 0 &&
           (domination_arrival_time > 0 || domination_changes_count > 0 || domination_journey_departure_time > 0);
  }

  bool is_equal(label& other) {
    if(arrival_time_ == other.arrival_time_ && journey_departure_time_ == other.journey_departure_time_ && changes_count_ == other.changes_count_
        && parent_station_ == other.parent_station_ && route_id_ == other.route_id_ && current_trip_id_ == other.current_trip_id_
        && stop_offset_ == other.stop_offset_ && footpath_duration_ == other.footpath_duration_) {
      return true;
    }
    return false;
  }

protected:
  inline int compare_to(int a, int b) {
    return (a > b) ? 1 : ((a == b) ? 0 : -1);
  }
};

struct route_label {
  // TODO: check
  const stop_time* trip_ = nullptr;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;

  route_stops_index parent_stop_ = invalid<route_stops_index>;
  time parent_journey_departure_time_ = invalid<time>;

  route_label() {
  }

  bool dominates(route_label& other) {
    return trip_ <= other.trip_;
  }

  bool is_equal(route_label& other) {
    return false;
  }
};

struct label_departure : public label<label_departure> {

  label_departure() {
  }

  label_departure(time departure_time, time arrival_time, size_t changes_count)
      : label(departure_time, arrival_time, changes_count){ }

  // to create labels for current round from labels from previous round for certain station
  label_departure(label_departure& parent_label, stop_id parent_station)
      : label(parent_label, parent_station) { }

  inline int arrival_time_rule(label& other) {
    return compare_to(other.arrival_time_, arrival_time_);
  }
  inline int journey_departure_time_rule(label& other) {
    return compare_to(journey_departure_time_, other.journey_departure_time_);
  }

  inline int changes_count_rule(label& other) {
    return compare_to(other.changes_count_, changes_count_);
  }

  inline int travel_duration_rule(label& other) {
    return compare_to(other.arrival_time_ - other.journey_departure_time_, arrival_time_ - journey_departure_time_);
  }

};

} // namespace motis::mcraptor

