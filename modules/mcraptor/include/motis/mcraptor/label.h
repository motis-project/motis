#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {

template <class T>
struct label {

  inline bool arrival_time_rule(label& other) {
    static_cast<T*>(this)->arrival_time_rule(other);
  }

  inline bool departure_time_rule(label& other) {
    static_cast<T*>(this)->departure_time_rule(other);
  }

  inline bool changes_count_rule(label& other) {
    static_cast<T*>(this)->changes_count_rule(other);
  }

  inline bool travel_duration_rule(label& other) {
    static_cast<T*>(this)->travel_duration_rule(other);
  }

public:
  // Parameters
  time departure_time_ = invalid<time>;
  size_t changes_count_ = invalid<size_t>;
  time arrival_time_ = invalid<time>;

  // Parent info
  stop_id parent_station_ = invalid<stop_id>;
  time parent_departure_time_ = invalid<time>;
  size_t parent_label_index_ = invalid<size_t>;

  // Current trip info
  route_id route_id_ = invalid<route_id>;
  trip_id current_trip_id_ = invalid<trip_id>;
  route_stops_index stop_offset_ = invalid<route_stops_index>;
  time footpath_duration_ = invalid<time>;

  label() {
  }

  label(time departure_time, time arrival_time, size_t changes_count) : departure_time_(departure_time),
                                                                     arrival_time_(arrival_time),
                                                                     changes_count_(changes_count) { }

  // to create labels for current round from labels from previous round for certain station
  label(label& parent_label, stop_id parent_station, size_t parent_index) : arrival_time_(parent_label.arrival_time_),
                                                                            changes_count_(parent_label.changes_count_ + 1),
                                                                            departure_time_(parent_label.departure_time_),
                                                                            parent_station_(parent_station),
                                                                            parent_label_index_(parent_index),
                                                                            parent_departure_time_(parent_label.arrival_time_) { }

  bool dominates(label& other) {
    return arrival_time_rule(other);
  }


  bool dominates_all(std::vector<label> labels) {
    for (label& l : labels) {
      if(!dominates(l)) {
        return false;
      }
    }
    return true;
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

  inline bool arrival_time_rule(label& other) {
    return arrival_time_ <= other.arrival_time_;
  }
  inline bool departure_time_rule(label& other) {
    return departure_time_ >= other.departure_time_;
  }

  inline bool changes_count_rule(label& other) {
    return changes_count_ <= other.changes_count_;
  }

  inline bool travel_duration_rule(label& other) {
    return (arrival_time_ - departure_time_) <= (other.arrival_time_ - other.departure_time_);
  }

};

struct label_arrival : public label<label_arrival> {

  label_arrival() {
  }

  label_arrival(time departure_time, time arrival_time, size_t changes_count)
      : label(departure_time, arrival_time, changes_count){ }

  // to create labels for current round from labels from previous round for certain station
  label_arrival(label_arrival& parent_label, stop_id parent_station, size_t parent_index)
      : label(parent_label, parent_station, parent_index) { }

  inline bool arrival_time_rule(label& other) {
    return arrival_time_ >= other.arrival_time_;
  }
  inline bool departure_time_rule(label& other) {
    return departure_time_ >= other.departure_time_;
  }

  inline bool changes_count_rule(label& other) {
    return changes_count_ <= other.changes_count_;
  }

  inline bool travel_duration_rule(label& other) {
    return (arrival_time_ - departure_time_) <= (other.arrival_time_ - other.departure_time_);
  }
};

} // namespace motis::mcraptor

