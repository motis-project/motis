#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "boost/graph/properties.hpp"

#include "motis-api/motis-api.h"

namespace motis::qa {

constexpr auto kMaxRating = std::numeric_limits<double>::max();
constexpr auto kMinRating = std::numeric_limits<double>::min();

using criterion_t = std::function<double(api::Itinerary const&)>;

// rate cmp in comparison to ref: positive value -> improvement
double rate(std::vector<api::Itinerary> const& cmp,
            std::vector<api::Itinerary> const& ref,
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

template <double Weight>
double walkingTime(api::Itinerary const& i) {
  return std::accumulate(begin(i.legs_), end(i.legs_), 0,[](auto const& a, auto const& b) { return a + (b.mode_ == api::ModeEnum::WALK ? b.duration_ : 0); }) * Weight;
}

}  // namespace criterion

}  // namespace motis::qa