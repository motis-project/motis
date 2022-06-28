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

  void merge(Label& label) {
    size_t removedLabels = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i].dominates(label)) return;
      if (label.dominates(labels[i])) {
        removedLabels++;
        continue;
      }
      labels[i - removedLabels] = labels[i];
    }
    labels.resize(labels.size() - removedLabels + 1);
    labels.back() = label;
    return;
  }

  inline bool isValid() const {
    return size() == 0;
  }
};

using BestBags = std::vector<Bag>;

} // namespace motis::mcraptor