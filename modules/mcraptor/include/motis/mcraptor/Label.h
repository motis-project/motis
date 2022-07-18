#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {


class Label {

  inline bool arrivalTimeRule(Label& other) {
    return arrivalTime <= other.arrivalTime;
  }

  inline bool departureTimeRule(Label& other) {
    return departureTime >= other.departureTime;
  }

  inline bool changesCountRule(Label& other) {
    return changesCount <= other.changesCount;
  }

  inline bool travelDurationRule(Label& other) {
    return (arrivalTime - departureTime) <= (other.arrivalTime - other.departureTime);
  }

public:
  // Parameters
  time departureTime = invalid<time>;
  size_t changesCount = invalid<size_t>;
  time arrivalTime = invalid<time>;

  // Parent info
  stop_id parentStation = invalid<stop_id>;
  time parentDepartureTime = invalid<time>;
  size_t parentLabelIndex = invalid<size_t>;

  // Current trip info
  route_id routeId = invalid<route_id>;
  trip_id current_trip_id = invalid<trip_id>;
  route_stops_index stop_offset = invalid<route_stops_index>;
  time foot_path_duration = invalid<time>;

  Label() {
  }

  Label(time departureTime, time arrivalTime, size_t changesCount) : departureTime(departureTime),
                                                                     arrivalTime(arrivalTime),
                                                                     changesCount(changesCount) { }

  // to create labels for current round from labels from previous round for certain station
  Label(Label& parentLabel, stop_id parentStation, size_t parentIndex) : arrivalTime(parentLabel.arrivalTime),
                                                                         changesCount(parentLabel.changesCount + 1),
                                                                         departureTime(parentLabel.departureTime),
                                                                         parentStation(parentStation),
                                                                         parentLabelIndex(parentIndex),
                                                                         parentDepartureTime(parentLabel.arrivalTime) { }

  bool dominates(Label& other) {
    return arrivalTimeRule(other);
  }


  bool dominatesAll(std::vector<Label> labels) {
    for (Label& l : labels) {
      if(!dominates(l)) {
        return false;
      }
    }
    return true;
  }
};

struct RouteLabel {

  RouteLabel() = default;

  // TODO: check
  const stop_time* trip = nullptr;
  trip_id current_trip_id = invalid<trip_id>;
  route_stops_index stop_offset = invalid<route_stops_index>;

  route_stops_index parentStop = invalid<route_stops_index>;
  size_t parentLabelIndex = invalid<size_t>;

  bool dominates(RouteLabel& otherLabel) {
    return trip <= otherLabel.trip;
  }
};


} // namespace motis::mcraptor

