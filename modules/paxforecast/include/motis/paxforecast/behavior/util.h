#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxforecast::behavior {

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

inline void only_keep_best_alternative(std::vector<float>& probabilities) {
  auto best_idx = 0ULL;
  auto best_prob = 0.0F;
  for (auto i = 0ULL; i < probabilities.size(); ++i) {
    auto const p = probabilities[i];
    if (p > best_prob) {
      best_idx = i;
      best_prob = p;
    }
    probabilities[i] = 0.0F;
  }
  probabilities[best_idx] = 1.0F;
}

}  // namespace motis::paxforecast::behavior
