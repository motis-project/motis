#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {


class Label {
//  typedef bool (*rule)(Label& label, Label& other);
//
//  static bool arrivalTimeRule(Label& label, Label& other) {
//    return label.arrivalTime <= other.arrivalTime;
//  }
//
//  static bool departureTimeRule(Label& label, Label& other) {
//    return label.departureTime >= other.departureTime;
//  }
//
//  static bool changesCountRule(Label& label, Label& other) {
//    return label.changesCount <= other.changesCount;
//  }


//  std::vector<rule> rules;

public:
  // Parameters
  time departureTime = invalid<time>;
  size_t changesCount = invalid<size_t>;
  time arrivalTime = invalid<time>;

  // Parent info
  stop_id parentStation = invalid<stop_id>;
  time parentDepartureTime = invalid<time>;
  size_t parentIndex = invalid<size_t>;
  route_id routeId = invalid<route_id>;


  // Constructor with rules
  Label() {
//    rules.push_back(arrivalTimeRule);
//    rules.push_back(departureTimeRule);
//    rules.push_back(changesCountRule);
  }

  Label(time departureTime, time arrivalTime, size_t changesCount) : departureTime(departureTime),
                                                                     arrivalTime(arrivalTime),
                                                                     changesCount(changesCount) { }

  // to create labels for current round from labels from previous round for certain station
  Label(Label& parentLabel, stop_id parentStation, size_t parentIndex) : arrivalTime(parentLabel.arrivalTime),
                                                                         parentStation(parentStation),
                                                                         parentIndex(parentIndex),
                                                                         parentDepartureTime(parentLabel.arrivalTime) { }

  bool dominates(Label& other) {
//    for (rule& r : rules) {
//      if(!r(*this, other)) {
//        return false;
//      }
//    }
//    return true;
  return arrivalTime <= other.arrivalTime;
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

  const stop_time* trip = nullptr;

  route_stops_index parentStop = invalid<route_stops_index>;
  size_t parentIndex = invalid<size_t>;

  bool dominates(RouteLabel& otherLabel) {
    return trip -> arrival_ <= otherLabel.trip -> arrival_ &&
           trip -> departure_ <= otherLabel.trip -> departure_;
  }
};


} // namespace motis::mcraptor

