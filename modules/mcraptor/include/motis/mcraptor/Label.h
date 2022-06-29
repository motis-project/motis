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
  time arrivalTime = invalid<time>;
  size_t changesCount = invalid<size_t>;

  // Constructor with rules
  Label() {
//    rules.push_back(arrivalTimeRule);
//    rules.push_back(departureTimeRule);
//    rules.push_back(changesCountRule);
  }

  Label(time departureTime, time arrivalTime, size_t changesCount) :
    departureTime(departureTime), arrivalTime(arrivalTime), changesCount(changesCount) { }

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


} // namespace motis::mcraptor

