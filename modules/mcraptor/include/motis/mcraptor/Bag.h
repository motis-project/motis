#pragma once

#include "Label.h"

namespace motis::mcraptor {

struct Bag {
  std::vector<Label> labels;

  Bag() {
    labels = std::vector<Label>();
  }

  inline Label& operator[](const size_t index) { return labels[index]; }

  inline const Label& operator[](const size_t i) const {
    return labels[i];
  }

  inline size_t size() const { return labels.size(); }

  void merge(Label& other) {
    if(dominates(other)) {
      return;
    }

    size_t removedLabels = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      if (other.dominates(labels[i])) {
        removedLabels++;
        continue;
      }
      labels[i - removedLabels] = labels[i];
    }
    labels.resize(labels.size() - removedLabels + 1);
    labels.back() = other;
    return;
  }

  void merge(Bag& other) {
    for(Label& otherLabel : other.labels) {
      merge(otherLabel);
    }
  }

  bool dominates(Label& other) {
    for(Label& label : labels) {
      if(label.dominates(other)) {
        return true;
      }
    }
    return false;
  }

  bool dominates(Bag& other) {
    if(!other.isValid() && this->isValid()) {
      return true;
    }

    for(Label& otherLabel : other.labels) {
      if(!dominates(otherLabel)) {
        return false;
      }
    }
    return true;
  }

  inline bool isValid() const {
    return size() != 0;
  }

  time getEarliestArrivalTime() {
    time res = invalid<time>;
    for(Label& label : labels) {
      res = std::min(res, label.arrivalTime);
    }
    return res;
  }
};

using BestBags = std::vector<Bag>;

} // namespace motis::mcraptor