#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {


class Label {
  typedef bool (*rule)(Label& label, Label& other);
  static bool arrivalTimeRule(Label& label, Label& other) {
    return label.arrivalTime <= other.arrivalTime;
  }

  static bool departureTimeRule(Label& label, Label& other) {
    return label.departureTime >= other.departureTime;
  }

  static bool changesCountRule(Label& label, Label& other) {
    return label.changesCount <= other.changesCount;
  }


  std::vector<rule> rules;

public:
  // Parameters
  time departureTime = invalid<time>;
  time arrivalTime = invalid<time>;
  size_t changesCount = invalid<size_t>;

  // Constructor with rules
  Label() {
    rules.push_back(arrivalTimeRule);
//    rules.push_back(departureTimeRule);
//    rules.push_back(changesCountRule);
  }

  Label(time departureTime, time arrivalTime, size_t changesCount) :
    departureTime(departureTime), arrivalTime(arrivalTime), changesCount(changesCount) { }

  bool dominates(Label& other) {
    bool d = true;
    for (rule& r : rules) {
      d &= r(*this, other);
    }
    return d;
  }

  bool dominatesAll(std::vector<Label> labels) {
    bool d = true;
    for (Label& l : labels) {
      d &= dominates(l);
    }
    return d;
  }

  bool dominatesAny(std::vector<Label> labels) {
    bool d = false;
    for (Label& l : labels) {
      d |= dominates(l);
    }
    return d;
  }



};


} // namespace motis::mcraptor

