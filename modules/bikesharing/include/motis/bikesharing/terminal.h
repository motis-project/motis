#pragma once

#include <ctime>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "motis/protocol/BikesharingCommon_generated.h"

namespace motis::bikesharing {

struct terminal {
  std::string uid_;
  double lat_ = 0.0, lng_ = 0.0;
  std::string name_;
};

struct terminal_snapshot : public terminal {
  int available_bikes_ = 0;
};

struct availability {
  double average_ = 0;
  double median_ = 0;
  double minimum_ = 0;
  double q90_ = 0;
  double percent_reliable_ = 0;
};

struct close_location {
  std::string id_;
  int duration_ = 0;
};

constexpr size_t kHoursPerDay = 24;
constexpr size_t kDaysPerWeek = 7;
constexpr size_t kBucketCount = kHoursPerDay * kDaysPerWeek;

template <typename T>
using hourly = std::array<T, kBucketCount>;
using hourly_availabilities = hourly<availability>;

size_t timestamp_to_bucket(std::time_t timestamp);

struct snapshot_merger {
  using hourly_buckets = hourly<std::vector<int>>;

  void add_snapshot(std::time_t, std::vector<terminal_snapshot> const&);

  std::pair<std::vector<terminal>, std::vector<hourly_availabilities>> merged();

  size_t snapshot_count_ = 0;
  std::map<std::string, terminal> terminals_;
  std::map<std::string, hourly_buckets> distributions_;
};

struct Availability;  // NOLINT
double get_availability(Availability const*, AvailabilityAggregator);

}  // namespace motis::bikesharing
