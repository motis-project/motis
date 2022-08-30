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

  inline L& operator[](const size_t index) { return labels_[index]; }

  inline const L& operator[](const size_t i) const {
    return labels_[i];
  }

  inline size_t size() const { return labels_.size(); }

  bool merge(L& other_label) noexcept {
    size_t removed_labels = 0;
    for (size_t i = 0; i < labels_.size(); i++) {
      if (labels_[i].dominates(other_label)) return false;
      if (other_label.dominates(labels_[i])) {
        removed_labels++;
        continue;
      }
      labels_[i - removed_labels] = labels_[i];
    }
    labels_.resize(labels_.size() - removed_labels + 1);
    labels_.back() = other_label;
    return true;
  }

  void merge_undominated(L& other_label) noexcept {
    if(dominates(other_label)){ //TODO why is it asserted in original?
//      std::cout << "mergeUndominated returns" << std::endl;
      return;
    }
    size_t removed_labels = 0;
    for (size_t i = 0; i < labels_.size(); i++) {
      if (other_label.dominates(labels_[i])) {
        removed_labels++;
        continue;
      }
      labels_[i - removed_labels] = labels_[i];
    }
    labels_.resize(labels_.size() - removed_labels + 1);
    labels_.back() = other_label;
  }

  // possibly delete
  void merge(bag& other_bag) {
    for(L& other_label : other_bag.labels_) {
      merge(other_label);
    }
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
    return size() != 0;
  }

  time get_earliest_arrival_time() {
    time res = invalid<time>;
    for(L& label : labels_) {
      res = std::min(res, label.arrival_time_);
    }
    return res;
  }
};

struct route_bag {

  std::vector<route_label> labels_;

  route_bag() {
    labels_ = std::vector<route_label>();
  }

  inline size_t size() const { return labels_.size(); }

  void merge(route_label& other_label) noexcept {
    size_t removed_labels = 0;
    for (size_t i = 0; i < labels_.size(); i++) {
      if (labels_[i].dominates(other_label)) {
        return;
      }
      if (other_label.dominates(labels_[i])) {
        removed_labels++;
        continue;
      }
      labels_[i - removed_labels] = labels_[i];
    }
    labels_.resize(labels_.size() - removed_labels + 1);
    labels_.back() = other_label;
  }

  inline route_label& operator[](const size_t index) { return labels_[index]; }

  inline const route_label& operator[](const size_t i) const {
    return labels_[i];
  }

};

//using BestBags = std::vector<Bag>;
//using bestRouteLabels = std::vector<Bag>;
//using bestTransferLabels = std::vector<Bag>;

/*struct TypesCollection {

  TypesCollection() {
    bestRouteLabels = std::vector<Bag>();
    bestTransferLabels = std::vector<Bag>();
    routesServingUpdatedStops = std::map<route_id, route_stops_index>();
    stopsUpdatedByRoute = std::vector<stop_id>();
    stopsUpdatedByTransfers = std::vector<stop_id>();
  }

  TypesCollection(stop_id stopCount, route_id routeCount) : bestRouteLabels(stopCount),
                                                            bestTransferLabels(stopCount),
                                                            stopsUpdatedByRoute(stopCount),
                                                            stopsUpdatedByTransfers(stopCount),
                                                            routesServingUpdatedStops() { };

public:
  std::vector<Bag> bestRouteLabels;
  std::vector<Bag> bestTransferLabels;

  // adapt to our parameters:
  // route_id and route_stop_index
  std::map<route_id, route_stops_index> routesServingUpdatedStops;
  std::vector<stop_id> stopsUpdatedByTransfers;
  std::vector<stop_id> stopsUpdatedByRoute;
};*/

} // namespace motis::mcraptor