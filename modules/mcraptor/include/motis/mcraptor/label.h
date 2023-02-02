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
  uint8_t changes_count_ = invalid<uint8_t>;
  time arrival_time_ = invalid<time>;

  // Parent info
  stop_id parent_station_ = invalid<stop_id>;
  size_t parent_label_index_ = invalid<size_t>;

  // Current trip info
  route_id route_id_ = invalid<route_id>;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;
  time footpath_duration_ = invalid<time>;

  raptor_round round_ = invalid<raptor_round>;

  label() {
  }

  label(time journey_departure_time, time arrival_time, size_t changes_count) : journey_departure_time_(journey_departure_time),
                                                                     arrival_time_(arrival_time),
                                                                     changes_count_(changes_count) { }

  // to create labels for current round from labels from previous round for certain station
  label(label& parent_label, stop_id parent_station, size_t parent_index) : arrival_time_(parent_label.arrival_time_),
                                                                            changes_count_(parent_label.changes_count_ + 1),
                                                                            journey_departure_time_(parent_label.journey_departure_time_),
                                                                            parent_station_(parent_station),
                                                                            parent_label_index_(parent_index){ }

  bool dominates(label& other) {
    int domination_arrival_time = arrival_time_rule(other);
    int domination_journey_departure_time = journey_departure_time_rule(other);
    int domination_changes_count = changes_count_rule(other);
    int domination_travel_duration = travel_duration_rule(other);

    // If equal and changes more or equal => dominate
//    if(domination_arrival_time == 0 && domination_journey_departure_time == 0) {
//      return domination_changes_count != -1;
//    }

    // If arrival time is earlier
    /*if (domination_arrival_time == 1) {
      // If more changes but duration is much less (see MAX_DIFF_FOR_LESS_TRANSFERS) => dominate
      if(domination_changes_count == -1 && domination_travel_duration >= 0) {
        time diff = (other.arrival_time_ - other.journey_departure_time_) - (arrival_time_ - journey_departure_time_);
        size_t diff_changes_count = this->changes_count_ - other.changes_count_;
        return diff > MAX_DIFF_FOR_LESS_TRANSFERS || diff_changes_count == 1;
      }
      // If arrival time is earlier AND journey departure time is earlier => do not dominate
      if(domination_changes_count == 0 && domination_journey_departure_time == -1) {
        return false;
      }
      return true;
    }
    else if (domination_arrival_time == 0) {
      // If arrival time is equal and departure time is later => dominate
      // return domination_journey_departure_time == 1;
      if(domination_changes_count == 1) {
        return true;
      }
      return domination_journey_departure_time == 1;
    }

    return false;*/

    return domination_arrival_time >= 0 && domination_changes_count >= 0 && domination_journey_departure_time >= 0 &&
           (domination_arrival_time > 0 || domination_changes_count > 0 || domination_journey_departure_time > 0);
  }

  bool dominates_all(std::vector<label> labels) {
    for (label& l : labels) {
      if(!dominates(l)) {
        return false;
      }
    }
    return true;
  }

protected:
  inline int compare_to(int a, int b) {
    return (a > b) ? 1 : ((a == b) ? 0 : -1);
  }
};

struct route_label {

  route_label() = default;

  // TODO: check
  const stop_time* trip_ = nullptr;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;

  route_stops_index parent_stop_ = invalid<route_stops_index>;
  size_t parent_label_index_ = invalid<size_t>;

  bool dominates(route_label& other_label) {
    return trip_ <= other_label.trip_;
  }
};

struct label_departure : public label<label_departure> {

  label_departure() {
  }

  label_departure(time departure_time, time arrival_time, size_t changes_count)
      : label(departure_time, arrival_time, changes_count){ }

  // to create labels for current round from labels from previous round for certain station
  label_departure(label_departure& parent_label, stop_id parent_station, size_t parent_index)
      : label(parent_label, parent_station, parent_index) { }

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

struct label_arrival : public label<label_arrival> {

  stop_id backward_parent_station = invalid<stop_id>;
  size_t backward_parent_label_index_ = invalid<size_t>;
  time departure_time_ = invalid<time>;


  label_arrival() {
  }

  label_arrival(time departure_time, time arrival_time, size_t changes_count)
      : label(departure_time, arrival_time, changes_count){ }

  // to create labels for current round from labels from previous round for certain station
  label_arrival(label_arrival& parent_label, stop_id parent_station, size_t parent_index)
      : label(parent_label, parent_station, parent_index) { }

  inline int arrival_time_rule(label& other) {
    return compare_to(arrival_time_, other.arrival_time_);
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

