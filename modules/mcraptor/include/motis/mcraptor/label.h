#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {

template <class T>
struct label {

  // Parameters
  time journey_departure_time_ = invalid<time>;
  size_t changes_count_ = invalid<size_t>;
  time arrival_time_ = invalid<time>;

  // Reconstructor
  stop_id current_target_ = invalid<stop_id>;

  // Parent info
  stop_id parent_station_ = invalid<stop_id>;

  // Current trip info
  route_id route_id_ = invalid<route_id>;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;
  time footpath_duration_ = invalid<time>;

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


  inline bool dominates(label& other) {
    return static_cast<T*>(this)->dominates(other);
  }

  inline bool dominates_supermega2(label& other) {
    return static_cast<T*>(this)->dominates(other);
  }

  bool is_equal(label& other) {
    return static_cast<T*>(this)->is_equal(other);
    }

  bool is_valid() {
    return static_cast<T*>(this)->is_valid();
  }

  bool is_in_range(time source_time_begin, time source_time_end) {
    return static_cast<T*>(this)->is_in_range(source_time_begin, source_time_end);
  }

  void out() {
    static_cast<T*>(this)->out();
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

  time parent_journey_arrival_time_ = invalid<time>;
  route_stops_index parent_stop_offset_ = invalid<route_stops_index>;
  time parent_arrival_time_ = invalid<time>;

  bool forward = true;

  route_label() {
  }

  bool dominates(route_label& other) {
    if(forward) {
      if(trip_ == other.trip_) {
        return parent_journey_departure_time_ >= other.parent_journey_departure_time_;
      }
      else {
        return trip_ < other.trip_;
      }
    }
    else {
      if(trip_ == other.trip_) {
        return parent_journey_arrival_time_ <= other.parent_journey_arrival_time_;
      }
      else {
        return trip_ > other.trip_;
      }
    }
  }

  bool dominates_supermega2(route_label& other) {
    return dominates(other);
  }

  bool is_equal(route_label& other) {
    return false;
  }
};

struct label_departure : public label<label_departure> {

  label_departure() {
  }

  label_departure(label& other) {
    // Parameters
    journey_departure_time_ = other.journey_departure_time_;
    changes_count_ = other.changes_count_;
    arrival_time_ = other.arrival_time_;

    // Reconstructor
    current_target_ = other.current_target_;

    // Parent info
    parent_station_ = other.parent_station_;

    // Current trip info
    route_id_ = other.route_id_;
    current_trip_id_ = other.current_trip_id_;
    stop_offset_ = other.stop_offset_;
    footpath_duration_ = other.footpath_duration_;
  }

  label_departure(time journey_departure_time, time arrival_time, size_t changes_count)
      : label(journey_departure_time, arrival_time, changes_count){ }

  // to create labels for current round from labels from previous round for certain station
  label_departure(label_departure& parent_label, stop_id parent_station)
      : label(parent_label, parent_station) { }

  bool dominates(label& other) {
    int domination_arrival_time = arrival_time_rule(other);
    int domination_journey_departure_time = journey_departure_time_rule(other);
    int domination_changes_count = changes_count_rule(other);

    return domination_arrival_time >= 0 && domination_changes_count >= 0 && domination_journey_departure_time >= 0 &&
           (domination_arrival_time > 0 || domination_changes_count > 0 || domination_journey_departure_time > 0);
  }

  bool dominates_supermega2(label& other) {
    return dominates(other);
  }

  bool is_equal(label& other) {
    if(arrival_time_ == other.arrival_time_ && journey_departure_time_ == other.journey_departure_time_ && changes_count_ == other.changes_count_) {
      return true;
    }
    return false;
  }

  bool is_valid() {
    return valid(arrival_time_) && valid(journey_departure_time_) && arrival_time_ >= journey_departure_time_;
  }

  bool is_in_range(time source_time_begin, time source_time_end) {
    return journey_departure_time_ >= source_time_begin &&
           journey_departure_time_ <= source_time_end;
  }

  void out() {
    std::cout << "TEST ----- : "
              << "; arr: " << arrival_time_
              << "; j dep: " << journey_departure_time_
              << "; changes: " << changes_count_ << std::endl;
  }
};

struct label_backward : public label<label_backward> {

  // Parameters
  time journey_arrival_time_ = invalid<time>;
  time departure_time_ = invalid<time>;

  // Parent info
  stop_id backward_parent_station_ = invalid<stop_id>;
  route_stops_index parent_stop_offset_ = invalid<route_stops_index>;
  time parent_arrival_time_ = invalid<time>;

  inline int journey_arrival_time_rule(label_backward& other) {
    return compare_to( other.journey_arrival_time_, journey_arrival_time_);
  }

  inline int departure_time_rule(label_backward& other) {
    return compare_to( departure_time_, other.departure_time_);
  }

  label_backward() {
  }

  label_backward(time journey_arrival_time, time departure_time, size_t changes_count)
      : label(invalid<time>, invalid<time>, changes_count){
    journey_arrival_time_ = journey_arrival_time;
    departure_time_ = departure_time;
  }

  // to create labels for current round from labels from previous round for certain station
  label_backward(label_backward& parent_label, stop_id backward_parent_station)
      : label(parent_label, invalid<stop_id>) {
    journey_arrival_time_ = parent_label.journey_arrival_time_;
    departure_time_ = parent_label.departure_time_;
    backward_parent_station_ = backward_parent_station;
    footpath_duration_ = 0;
    parent_arrival_time_ = parent_label.departure_time_;
  }

  bool dominates(label_backward& other) {
    int domination_departure_time = departure_time_rule(other);
    int domination_journey_arrival_time = journey_arrival_time_rule(other);
    int domination_changes_count = changes_count_rule(other);

    return domination_departure_time >= 0 && domination_changes_count >= 0 && domination_journey_arrival_time >= 0 &&
           (domination_departure_time > 0 || domination_changes_count > 0 || domination_journey_arrival_time > 0);
  }

  bool dominates_supermega2(label& other) {
    int domination_arrival_time = arrival_time_rule(other);
    int domination_journey_departure_time = journey_departure_time_rule(other);
    int domination_changes_count = changes_count_rule(other);

    return domination_arrival_time >= 0 && domination_changes_count >= 0 && domination_journey_departure_time >= 0 &&
           (domination_arrival_time > 0 || domination_changes_count > 0 || domination_journey_departure_time > 0);
  }

  bool is_equal(label_backward& other) {
    if(departure_time_ == other.departure_time_ && journey_arrival_time_ == other.journey_arrival_time_ && changes_count_ == other.changes_count_) {
      return true;
    }
    return false;
  }

  bool is_valid() {
    return valid(departure_time_) && valid(journey_arrival_time_) && departure_time_ <= journey_arrival_time_;
  }

  bool is_in_range(time source_time_begin, time source_time_end) {
//    std::cout << "RANGE: " << source_time_begin << " - " << source_time_end << "; value: " << journey_arrival_time_ << std::endl;
    return journey_arrival_time_ >= source_time_begin &&
           journey_arrival_time_ <= source_time_end;
  }

  void out() {
    std::cout << "TEST ----- : "
              << "; dep: " << departure_time_
              << "; arr: " << journey_arrival_time_
              << "; changes: " << changes_count_ << std::endl;
  }
};

} // namespace motis::mcraptor

