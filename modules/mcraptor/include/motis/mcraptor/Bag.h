#pragma once

#include <map>
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

  bool merge(Label& otherLabel) noexcept {
    size_t removedLabels = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i].dominates(otherLabel)) return false;
      if (otherLabel.dominates(labels[i])) {
        removedLabels++;
        continue;
      }
      labels[i - removedLabels] = labels[i];
    }
    labels.resize(labels.size() - removedLabels + 1);
    labels.back() = otherLabel;
    return true;
  }

  void mergeUndominated(Label& otherLabel) noexcept {
    if(dominates(otherLabel)){ //TODO why is it asserted in original?
//      std::cout << "mergeUndominated returns" << std::endl;
      return;
    }
    size_t removedLabels = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      if (otherLabel.dominates(labels[i])) {
        removedLabels++;
        continue;
      }
      labels[i - removedLabels] = labels[i];
    }
    labels.resize(labels.size() - removedLabels + 1);
    labels.back() = otherLabel;
  }

  // possibly delete
  void merge(Bag& otherBag) {
    for(Label& otherLabel : otherBag.labels) {
      merge(otherLabel);
    }
  }

  bool dominates(Label& otherLabel) {
    for(Label& label : labels) {
      if(label.dominates(otherLabel)) {
        return true;
      }
    }
    return false;
  }

  bool dominates(Bag& otherBag) {
    if(!otherBag.isValid() && this->isValid()) {
      return true;
    }

    for(Label& label : otherBag.labels) {
      if(!dominates(label)) {
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

struct RouteBag {

  std::vector<RouteLabel> labels;

  RouteBag() {
    labels = std::vector<RouteLabel>();
  }

  inline size_t size() const { return labels.size(); }

  void merge(RouteLabel& otherLabel) noexcept {
    size_t removedLabels = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i].dominates(otherLabel)) {
        return;
      }
      if (otherLabel.dominates(labels[i])) {
        removedLabels++;
        continue;
      }
      labels[i - removedLabels] = labels[i];
    }
    labels.resize(labels.size() - removedLabels + 1);
    labels.back() = otherLabel;
  }

  inline RouteLabel& operator[](const size_t index) { return labels[index]; }

  inline const RouteLabel& operator[](const size_t i) const {
    return labels[i];
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