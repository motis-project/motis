#include "motis/odm/mixer.h"

#include "utl/overloaded.h"

#include "nigiri/special_stations.h"

#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;

// journey cost
static auto const kWalkCost = std::vector<cost_threshold>{{0, 1}, {15, 11}};
static auto const kTaxiCost = std::vector<cost_threshold>{{0, 59}, {1, 13}};
static auto const kTransferCost = std::vector<cost_threshold>{{0, 15}};
static constexpr auto const kDirectTaxiFactor = 1.3;
static constexpr auto const kDirectTaxiConstant = 27;

// domination
static constexpr auto const kTravelTimeWeight = 1.5;
static constexpr auto const kDistanceWeight = 0.07;
static constexpr auto const kDistanceExponent = 1.5;

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
  return tally(j.transfers_, kTransferCost);
};

std::int32_t mixer::distance(n::routing::journey const& a,
                             n::routing::journey const& b) const {
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
};

void mixer::cost_domination(
    n::pareto_set<n::routing::journey> const& pt_journeys,
    std::vector<n::routing::journey>& odm_journeys) const {

  auto const leg_cost = [](auto const& leg) {
    return std::visit(
        utl::overloaded{[](n::routing::journey::run_enter_exit const& ree
                           [[maybe_unused]]) { return std::int32_t{0}; },
                        [](n::footpath const& fp) {
                          return tally(fp.duration().count(), kWalkCost);
                        },
                        [](n::routing::offset const& o) {
                          if (o.transport_mode_id_ == kODM) {
                            return tally(o.duration().count(), kTaxiCost);
                          } else if (o.transport_mode_id_ == kWalk) {
                            return tally(o.duration().count(), kWalkCost);
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

  auto const is_direct_taxi = [](auto const& j) {
    return j.legs_.size() == 1 &&
           j.legs_.front().from_ ==
               get_special_station(n::special_station::kStart) &&
           j.legs_.front().to_ ==
               get_special_station(n::special_station::kEnd) &&
           std::holds_alternative<n::routing::offset>(j.legs_.front().uses_) &&
           std::get<n::routing::offset>(j.legs_.front().uses_)
                   .transport_mode_id_ == kODM;
  };

  auto const cost = [&](auto const& j) {
    return (leg_cost(j.legs_.front()) +
            (j.legs_.size() > 1 ? leg_cost(j.legs_.back()) : 0) +
            pt_time(j).count() + transfer_cost(j)) *
               (is_direct_taxi(j) ? kDirectTaxiFactor : 1) +
           (is_direct_taxi(j) ? kDirectTaxiConstant : 0);
  };

  auto const is_dominated = [&](auto const& odm_journey) {
    auto const dominates = [&](auto const& pt_journey) {
      auto const protection =
          kTravelTimeWeight *
              (static_cast<double>(pt_journey.travel_time().count()) /
               static_cast<double>(pt_journey.travel_time().count())) +
          kDistanceWeight *
              std::pow(distance(pt_journey, odm_journey), kDistanceExponent);
      return cost(pt_journey) + protection < cost(odm_journey);
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
    auto const is_taxi_leg = [](auto const& l) {
      return std::holds_alternative<n::routing::offset>(l.uses_) &&
             std::get<n::routing::offset>(l.uses_).transport_mode_id_ == kODM;
    };

    return (is_taxi_leg(j.legs_.front())
                ? std::get<n::routing::offset>(j.legs_.front().uses_)
                      .duration()
                      .count()
                : 0) +
           ((j.legs_.size() > 1 && is_taxi_leg(j.legs_.back()))
                ? std::get<n::routing::offset>(j.legs_.back().uses_)
                      .duration()
                      .count()
                : 0);
  };

  auto const is_dominated = [&](auto const& b) {
    auto const dominates = [&](auto const& a) {
      auto const protection =
          kDistanceWeight * std::pow(distance(a, b), kDistanceExponent);
      return cost(b) / taxi_time(a) > (cost(a) + protection) / taxi_time(b);
    };

    return std::any_of(begin(odm_journeys), end(odm_journeys), dominates);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void mixer::mix(n::pareto_set<n::routing::journey> const& pt_journeys,
                std::vector<n::routing::journey>& odm_journeys) const {
  cost_domination(pt_journeys, odm_journeys);
  productivity_domination(odm_journeys);
  odm_journeys.append_range(pt_journeys);
}

}  // namespace motis::odm