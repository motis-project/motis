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
                                 .count()) *
         Weight;
}

template <double Weight>
double transfers(api::Itinerary const& i) {
  return static_cast<double>(i.transfers_) * Weight;
}

template <double Weight>
double walking_time(api::Itinerary const& i) {
  auto walking_time = std::int64_t{0};
  for (auto const& l : i.legs_) {
    if (l.mode_ == api::ModeEnum::WALK) {
      walking_time += l.duration_;
    }
  }
  return static_cast<double>(walking_time) * Weight;
}

constexpr auto kDefaultStartTime = start_time<1.0>;
constexpr auto kDefaultEndTime = end_time<1.0>;
constexpr auto kDefaultTransfers = transfers<30.0>;
constexpr auto kDefaultWalkingTime = walking_time<1.0 / 60.0>;

}  // namespace criterion

auto const kStartEndTransfer = std::vector<criterion_t>{
    criterion::kDefaultStartTime, criterion::kDefaultEndTime,
    criterion::kDefaultTransfers};
auto const kStartEndTransferWalk = std::vector<criterion_t>{
    criterion::kDefaultStartTime, criterion::kDefaultEndTime,
    criterion::kDefaultTransfers, criterion::kDefaultWalkingTime};

}  // namespace motis::qa