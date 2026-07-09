#include "motis/qa.h"

namespace motis::qa {

double set_improvement(
    std::vector<api::Itinerary> const& ref,
    std::vector<api::Itinerary> const& uut,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  static std::vector<std::reference_wrapper<api::Itinerary>> a;
  static std::vector<std::reference_wrapper<api::Itinerary>> b;
  a.clear();
  b.clear();
  a.emplace_back(ref.begin(), ref.end());
  b.emplace_back(uut.begin(), uut.end());

  auto impr = double{0.0};

  while (!a.empty()) {
    auto max_impr_a = kMinRating;
    auto max_a
  }
}

double rate(
    std::vector<api::Itinerary> const& ref,
    std::vector<api::Itinerary> const& uut,
    std::vector<std::function<double(api::Itinerary const&)>> const& criteria) {
  if (ref.empty() && uut.empty()) {
    return double{0.0};
  }
  if (ref.empty()) {
    return kMaxRating;
  }
  if (uut.empty()) {
    return kMinRating;
  }

  auto const LR = set_improvement(ref, uut, criteria);
  auto const RL = set_improvement(uut, ref, criteria);
  return LR - RL;
}

}  // namespace motis::qa