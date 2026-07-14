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
  return static_cast<double>(-i.startTime_.get_unixtime_seconds()) * Weight;
}

template <double Weight>
double end_time(api::Itinerary const& i) {
  return static_cast<double>(i.endTime_.get_unixtime_seconds()) * Weight;
}

template <double Weight>
double transfers(api::Itinerary const& i) {
  return static_cast<double>(i.transfers_) * Weight;
}

} // namespace criteria

inline double rate_classic(std::vector<api::Itinerary> const& a,
            std::vector<api::Itinerary> const& b) {
  static auto const criteria = std::vector<criterion_t>{criterion::start_time<1.0>, criterion::end_time<1.0>, criterion::transfers<30.0>};
  return rate(a,b, criteria);
}

}  // namespace motis::qa