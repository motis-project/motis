#pragma once

#include "motis/core/schedule/time.h"
#include "vector"

namespace motis::mcraptor {

class Label;
typedef bool (*rule)(Label& label, Label& other);
bool arrivalTimeRule(Label& label, Label& other);
bool departureTimeRule(Label& label, Label& other);
bool changesCountRule(Label& label, Label& other);

class Label {
  std::vector<rule> rules;

public:
  // Parameters
  time departureTime = invalid<time>;
  time arrivalTime = invalid<time>;
  size_t changesCount = invalid<size_t>;

  // Constructor with rules
  Label() {
    rules.push_back(arrivalTimeRule);
    rules.push_back(departureTimeRule);
    rules.push_back(changesCountRule);
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
};

bool arrivalTimeRule(Label& label, Label& other) {
  return label.arrivalTime <= other.arrivalTime;
}

bool departureTimeRule(Label& label, Label& other) {
  return label.departureTime >= other.departureTime;
}

bool changesCountRule(Label& label, Label& other) {
  return label.changesCount <= other.changesCount;
}

} // namespace motis::mcraptor

