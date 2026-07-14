#pragma once

#include <functional>
#include <vector>

#include "boost/graph/properties.hpp"

#include "motis-api/motis-api.h"

namespace motis::qa {

constexpr auto kMaxRating = std::numeric_limits<double>::max();
constexpr auto kMinRating = std::numeric_limits<double>::min();

using criterion_t = std::function<double(api::Itinerary const&)>;

double improvement(
    api::Itinerary const&,
    api::Itinerary const&,
    std::vector<criterion_t> const&);

double min_improvement(
    api::Itinerary const*,
    std::vector<api::Itinerary const*> const&,
    std::vector<criterion_t> const&);

double set_improvement(
    std::vector<api::Itinerary> const&,
    std::vector<api::Itinerary> const&,
    std::vector<criterion_t> const&);

double rate(std::vector<api::Itinerary> const&,
            std::vector<api::Itinerary> const&,
            std::vector<criterion_t> const&);

namespace criterion {

template <double Weight>
double start_time(api::Itinerary const& i) {
  return static_cast<double>(-std::chrono::round<std::chrono::minutes>(
                                  i.startTime_.time_.time_since_epoch())
                                  .count()) *
         Weight;
}

template <double Weight>
double end_time(api::Itinerary const& i) {
  return static_cast<double>(std::chrono::round<std::chrono::minutes>(
                                  i.endTime_.time_.time_since_epoch())
                                  .count()) * Weight;
}

template <double Weight>
double transfers(api::Itinerary const& i) {
  return static_cast<double>(i.transfers_) * Weight;
}

}  // namespace criterion

}  // namespace motis::qa