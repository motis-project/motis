#pragma once

#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "geo/latlng.h"

#include "motis/core/schedule/connection.h"

namespace motis {

namespace loader {

struct Schedule;  // NOLINT

}  // namespace loader

namespace path {

struct station_seq {
  friend bool operator<(station_seq const& lhs, station_seq const& rhs) {
    return std::tie(lhs.station_ids_, lhs.coordinates_) <
           std::tie(rhs.station_ids_, rhs.coordinates_);
  }

  friend bool operator==(station_seq const& lhs, station_seq const& rhs) {
    return std::tie(lhs.station_ids_, lhs.coordinates_) ==
           std::tie(rhs.station_ids_, rhs.coordinates_);
  }

  std::vector<std::string> station_ids_;
  std::vector<std::string> station_names_;
  std::vector<geo::latlng> coordinates_;

  std::set<service_class> classes_;
};

std::vector<station_seq> load_station_sequences(motis::loader::Schedule const*);

}  // namespace path
}  // namespace motis
