#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"

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

inline void only_keep_best_alternative(std::vector<float>& probabilities,
                                       float const best_probability = 1.0F) {
  if (probabilities.empty()) {
    return;
  }
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
  probabilities[best_idx] = best_probability;
}

inline std::vector<float> calc_new_probabilites(
    float const base_prob, std::vector<alternative> const& alts,
    float const threshold) {
  if (alts.empty()) {
    return {};
  }
  auto probs = utl::to_vec(
      alts, [&](auto const& alt) { return base_prob * alt.pick_probability_; });
  auto total_sum = 0.0F;
  auto kept_sum = 0.0F;
  auto rescale = false;
  for (auto& p : probs) {
    total_sum += p;
    if (p != 0.0F && p < threshold) {
      p = 0;
      rescale = true;
    } else {
      kept_sum += p;
    }
  }
  if (rescale && kept_sum != 0.0F) {
    auto const scale = kept_sum / total_sum;
    for (auto& p : probs) {
      p /= scale;
    }
  } else if (kept_sum == 0.0F) {
    probs = utl::to_vec(alts,
                        [&](auto const& alt) { return alt.pick_probability_; });
    only_keep_best_alternative(probs, base_prob);
  }
  return probs;
}

}  // namespace motis::paxforecast::behavior
