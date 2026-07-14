#include "motis/qa.h"

#include <cmath>

namespace motis::qa {

constexpr auto kP = double{30.0};
constexpr auto kQ = double{0.1};

double improvement(
    api::Itinerary const& a,
    api::Itinerary const& b,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  auto dist = double{0.0};
  auto impr = double{0.0};

  for (auto const& criterion : criteria) {
    auto const crit_dist = criterion(a) - criterion(b);
    dist += std::pow(crit_dist, 2);
    if (crit_dist < 0) {
      impr += std::pow(crit_dist, 2);
    }
  }

  dist = std::sqrt(dist);
  impr = std::sqrt(impr);

  if (impr == 0.0) {
    return 0.0;
  }

  return std::log2(std::pow(impr, 2) / dist) *
         (std::atan(kP * (dist - kQ)) + std::numbers::pi / 2.0);
}

double min_improvement(
    api::Itinerary const* i,
    std::vector<api::Itinerary const*> const& js,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  auto min_impr = kMaxRating;

  for (auto const j : js) {
    auto const impr = improvement(*i, *j, criteria);
    if (impr < min_impr) {
      min_impr = impr;
    }
  }

  return min_impr;
}

double set_improvement(
    std::vector<api::Itinerary> const& a,
    std::vector<api::Itinerary> const& b,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  static std::vector<api::Itinerary const*> ap;
  static std::vector<api::Itinerary const*> bp;

  auto const reset = [](std::vector<api::Itinerary> const& v,
                        std::vector<api::Itinerary const*>& p) {
    p.clear();
    for (auto const& i : v) {
      p.emplace_back(&i);
    }
  };

  reset(a, ap);
  reset(b, bp);

  auto impr = double{0.0};

  while (!ap.empty()) {
    auto max_impr_a = kMinRating;
    auto max_a = 0U;

    for (auto i = 0U; i != ap.size(); ++i) {
      auto const min_impr = min_improvement(ap[i], bp, criteria);
      if (min_impr > max_impr_a) {
        max_impr_a = min_impr;
        max_a = i;
      }
    }

    impr += max_impr_a;

    std::swap(ap[max_a], ap.back());
    bp.emplace_back(ap.back());
    ap.pop_back();
  }

  return impr;
}

double rate(
    std::vector<api::Itinerary> const& a,
    std::vector<api::Itinerary> const& b,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  if (a.empty() && b.empty()) {
    return 0.0;
  }
  if (a.empty()) {
    return kMinRating;
  }
  if (b.empty()) {
    return kMaxRating;
  }

  auto const LR = set_improvement(a, b, criteria);
  auto const RL = set_improvement(b, a, criteria);
  return LR - RL;
}

}  // namespace motis::qa