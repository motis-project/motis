#pragma once

#include <map>
#include "label.h"

namespace motis::mcraptor {
template <class L>
struct bag {
  std::vector<L> labels_;

  bag() {
    labels_ = std::vector<L>();
  }

  L& get_fastest_label(motis::time next_departure_time, L& def) {
    if (labels_.empty()) {
      return def;
    }

    motis::time min_diff = invalid<motis::time>;
    L* closest_label = nullptr;
    for (L& label : labels_) {
      if (label.arrival_time_ <= next_departure_time) {
        motis::time current_diff = next_departure_time - label.journey_departure_time_;
        if (current_diff < min_diff) {
          closest_label = &label;
          min_diff = current_diff;
        }
      }
    }

    return closest_label == nullptr ? def : *closest_label;
  }

  L& get_fastest_backward_label(motis::time prev_arrival_time, L& def) {
    if (labels_.empty()) {
      return def;
    }

    motis::time min_diff = invalid<motis::time>;
    L* closest_label = nullptr;
    for (L& label : labels_) {
      if (label.departure_time_ >= prev_arrival_time) {
        motis::time current_diff = label.journey_arrival_time_ - prev_arrival_time;
        if (current_diff < min_diff) {
          closest_label = &label;
          min_diff = current_diff;
        }
      }
    }

    return closest_label == nullptr ? def : *closest_label;
  }
  bool merge(L& other_label, bool on_reconstruct = false) noexcept {
    if(on_reconstruct) {
      size_t removed_labels = 0;
      size_t labels_size = labels_.size();
      for (size_t i = 0; i < labels_size; i++) {
        if(labels_[i].is_equal(other_label)) return !on_reconstruct;
        if (labels_[i].dominates_supermega2(other_label)) return false;
        if (other_label.dominates_supermega2(labels_[i])) {
          removed_labels++;
          continue;
        }
        labels_[i - removed_labels] = labels_[i];
      }
      labels_.resize(labels_size - removed_labels + 1);
      labels_.back() = other_label;
      return true;
    }


    size_t removed_labels = 0;
    size_t labels_size = labels_.size();
    for (size_t i = 0; i < labels_size; i++) {
      if(labels_[i].is_equal(other_label)) return !on_reconstruct;
      if (labels_[i].dominates(other_label)) return false;
      if (other_label.dominates(labels_[i])) {
        removed_labels++;
        continue;
      }
      labels_[i - removed_labels] = labels_[i];
    }
    labels_.resize(labels_size - removed_labels + 1);
    labels_.back() = other_label;
    return true;
  }

  void merge_undominated(L& other_label) noexcept {
    size_t removed_labels = 0;
    size_t labels_size = labels_.size();
    for (size_t i = 0; i < labels_size; i++) {
      if(labels_[i].is_equal(other_label)) return;
      if (other_label.dominates(labels_[i])) {
        removed_labels++;
        continue;
      }
      labels_[i - removed_labels] = labels_[i];
    }
    labels_.resize(labels_size - removed_labels + 1);
    labels_.back() = other_label;
  }

  bool dominates(L& other_label) {
    for(L& label : labels_) {
      if(label.dominates(other_label)) {
        return true;
      }
    }
    return false;
  }

  bool dominates(bag& other_bag) {
    if(!other_bag.is_valid() && this->is_valid()) {
      return true;
    }

    for(L& label : other_bag.labels_) {
      if(!dominates(label)) {
        return false;
      }
    }
    return true;
  }

  inline bool is_valid() const {
    return !labels_.empty();
  }
};


} // namespace motis::mcraptor