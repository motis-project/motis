#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "utl/enumerate.h"
#include "utl/verify.h"

namespace motis::paxforecast::behavior {

inline std::optional<std::size_t> get_recommended_alternative(
    motis::paxmon::passenger_group const& grp,
    std::vector<measures::please_use> const& announcements) {
  for (auto const& announcement : announcements) {
    if (announcement.direction_station_id_ ==
        grp.compact_planned_journey_.destination_station_id()) {
      return {announcement.alternative_id_};
    }
  }
  return {};
}

template <typename T>
std::vector<std::size_t> min_indices(std::vector<T> const& v) {
  auto best_score = std::numeric_limits<T>::max();
  auto selected = std::vector<std::size_t>{};
  for (auto const [i, score] : utl::enumerate(v)) {
    if (score < best_score) {
      best_score = score;
      selected.clear();
      selected.push_back(i);
    } else if (score == best_score) {
      selected.push_back(i);
    }
  }
  utl::verify(selected.empty() == v.empty(), "min_indices failed");
  return selected;
}

template <typename T>
std::vector<std::size_t> max_indices(std::vector<T> const& v) {
  auto best_score = std::numeric_limits<T>::min();
  auto selected = std::vector<std::size_t>{};
  for (auto const [i, score] : utl::enumerate(v)) {
    if (score > best_score) {
      best_score = score;
      selected.clear();
      selected.push_back(i);
    } else if (score == best_score) {
      selected.push_back(i);
    }
  }
  utl::verify(selected.empty() == v.empty(), "max_indices failed");
  return selected;
}

}  // namespace motis::paxforecast::behavior
