#include "motis/odm/mixer.h"

#include "utl/overloaded.h"

#include "nigiri/special_stations.h"

#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;

std::int32_t tally(std::int32_t const x,
                   std::vector<cost_threshold> const& ct) {
  auto acc = std::int32_t{0};
  for (auto i = 0U; i < ct.size() && ct[i].threshold_ < x; ++i) {
    auto const valid_until = i + 1U == ct.size()
                                 ? std::numeric_limits<std::int32_t>::max()
                                 : ct[i + 1U].threshold_;
    acc += (std::min(x, valid_until) - ct[i].threshold_) * ct[i].cost_;
  }
  return acc;
}

std::int32_t mixer::transfer_cost(n::routing::journey const& j) const {
  return tally(j.transfers_, transfer_cost_);
}

std::int32_t distance(n::routing::journey const& a,
                      n::routing::journey const& b) {
  auto const overtakes = [](auto const& x, auto const& y) {
    return x.departure_time() > y.departure_time() &&
           x.arrival_time() < y.arrival_time();
  };

  return overtakes(a, b) || overtakes(b, a)
             ? 0
             : std::min(
                   std::chrono::abs(a.departure_time() - b.departure_time()),
                   std::chrono::abs(a.arrival_time() - b.arrival_time()))
                   .count();
}

void mixer::cost_domination(
    n::pareto_set<n::routing::journey> const& pt_journeys,
    std::vector<n::routing::journey>& odm_journeys) const {

  auto const leg_cost = [&](auto const& leg) {
    return std::visit(
        utl::overloaded{[](n::routing::journey::run_enter_exit const&) {
                          return std::int32_t{0};
                        },
                        [&](n::footpath const& fp) {
                          return tally(fp.duration().count(), walk_cost_);
                        },
                        [&](n::routing::offset const& o) {
                          if (o.transport_mode_id_ == kODM) {
                            return tally(o.duration().count(), taxi_cost_);
                          } else if (o.transport_mode_id_ == kWalk) {
                            return tally(o.duration().count(), walk_cost_);
                          }
                          utl::verify(o.transport_mode_id_ == kODM ||
                                          o.transport_mode_id_ == kWalk,
                                      "unknown transport mode");
                          return std::int32_t{0};
                        }},
        leg.uses_);
  };

  auto const pt_time = [](auto const& j) {
    auto const leg_duration = [](auto const& l) {
      return std::visit(
          utl::overloaded{
              [](n::routing::journey::run_enter_exit const& ree
                 [[maybe_unused]]) { return n::duration_t{0}; },
              [](n::footpath const& fp) { return fp.duration(); },
              [](n::routing::offset const& o) { return o.duration(); }},
          l.uses_);
    };
    return j.travel_time() - leg_duration(j.legs_.front()) -
           ((j.legs_.size() > 1) ? leg_duration(j.legs_.back())
                                 : n::duration_t{0});
  };

  auto const cost = [&](auto const& j) {
    return (leg_cost(j.legs_.front()) +
            (j.legs_.size() > 1 ? leg_cost(j.legs_.back()) : 0) +
            pt_time(j).count() + transfer_cost(j));
  };

  auto const is_dominated = [&](auto const& odm_journey) {
    auto const dominates = [&](auto const& pt_journey) {
      auto const alpha_term =
          alpha_ *
          (static_cast<double>(pt_journey.travel_time().count()) /
           static_cast<double>(odm_journey.travel_time().count())) *
          distance(pt_journey, odm_journey);
      return cost(pt_journey) + alpha_term < cost(odm_journey);
    };

    return std::any_of(begin(pt_journeys), end(pt_journeys), dominates);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void mixer::productivity_domination(
    std::vector<n::routing::journey>& odm_journeys) const {
  auto const cost = [&](auto const& j) -> double {
    return j.travel_time().count() + transfer_cost(j);
  };

  auto const taxi_time = [](n::routing::journey const& j) -> double {
    return (is_odm_leg(j.legs_.front())
                ? std::get<n::routing::offset>(j.legs_.front().uses_)
                      .duration()
                      .count()
                : 0) +
           ((j.legs_.size() > 1 && is_odm_leg(j.legs_.back()))
                ? std::get<n::routing::offset>(j.legs_.back().uses_)
                      .duration()
                      .count()
                : 0);
  };

  auto const is_dominated = [&](auto const& b) {
    auto const dominates_b = [&](auto const& a) {
      auto const prod_a = cost(b) / taxi_time(a);
      auto const prod_b = (cost(a) + beta_ * distance(a, b)) / taxi_time(b);
      return prod_a > prod_b;
    };
    return utl::any_of(odm_journeys, dominates_b);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void mixer::mix(n::pareto_set<n::routing::journey> const& pt_journeys,
                std::vector<n::routing::journey>& odm_journeys) const {
  cost_domination(pt_journeys, odm_journeys);
  productivity_domination(odm_journeys);
  for (auto const& j : pt_journeys) {
    odm_journeys.emplace_back(j);
  }
  utl::sort(odm_journeys, [](auto const& a, auto const& b) {
    return a.departure_time() < b.departure_time();
  });
}

}  // namespace motis::odm