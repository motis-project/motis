#include "motis/odm/mix.h"

#include "utl/overloaded.h"

namespace motis::odm {

// journey cost
static auto const kWalkCost = std::vector<cost_threshold>{{0, 1}, {15, 11}};
static auto const kTaxiCost = std::vector<cost_threshold>{{0, 59}, {1, 13}};
static auto const kTransferCost = std::vector<cost_threshold>{{0, 15}};
static constexpr auto const kDirectTaxiFactor = 1.3;
static constexpr auto const kDirectTaxiConstant = 27;

// cost domination
static constexpr auto const kTravelTimeWeight = 1.5;
static constexpr auto const kDistanceWeight = 0.07;
static constexpr auto const kDistanceExponent = 1.5;

// productivity domination

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

void cost_domination(n::pareto_set<n::routing::journey> const& pt_journeys,
                     std::vector<n::routing::journey>& odm_journeys) {

  auto const leg_cost = [](n::routing::journey::leg const& leg) {
    return std::visit(
        utl::overloaded{[](n::routing::journey::run_enter_exit const& ree) {
                          return std::int32_t{0};
                        },
                        [](n::footpath const& fp) {
                          return tally(fp.duration().count(), kWalkCost);
                        },
                        [](n::routing::offset const& o) {
                          return o.transport_mode_id_ == offset_mode::kTaxi
                                     ? tally(o.duration().count(), kTaxiCost)
                                     : std::int32_t{0};
                        }},
        leg.uses_);
  };

  auto const transfer_cost = [](n::routing::journey const& j) {
    return tally(j.transfers_, kTransferCost);
  };

  auto const pt_time = [](n::routing::journey const& j) {
    auto const leg_duration = [](n::routing::journey::leg const& l) {
      return std::visit(
          utl::overloaded{
              [](n::routing::journey::run_enter_exit const& ree) {
                return n::duration_t{0};
              },
              [](n::footpath const& fp) { return fp.duration(); },
              [](n::routing::offset const& o) { return o.duration(); }},
          l.uses_);
    };
    return (j.arrival_time() - j.departure_time()) -
           leg_duration(j.legs_.front()) -
           ((j.legs_.size() > 1) ? leg_duration(j.legs_.back())
                                 : n::duration_t{0});
  };

  auto const is_direct_taxi = [](n::routing::journey const& j) {
    return j.legs_.size() == 1 &&
           std::holds_alternative<n::routing::offset>(j.legs_.front().uses_) &&
           std::get<n::routing::offset>(j.legs_.front().uses_)
                   .transport_mode_id_ == offset_mode::kTaxi;
  };

  auto const cost = [&](n::routing::journey const& j) {
    auto const direct_taxi = is_direct_taxi(j);
    return (leg_cost(j.legs_.front()) +
            (j.legs_.size() > 1 ? leg_cost(j.legs_.back()) : 0) +
            pt_time(j).count() + transfer_cost(j)) *
               (direct_taxi ? kDirectTaxiFactor : 1) +
           (direct_taxi ? kDirectTaxiConstant : 0);
  };

  auto const distance = [](n::routing::journey const& a,
                           n::routing::journey const& b) {
    auto const overtakes = [](n::routing::journey const& x,
                              n::routing::journey const& y) {
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

  auto const cost_dominates = [&](n::routing::journey const& a,
                                  n::routing::journey const& b) {
    auto const protection = ;
    return cost(a) + protection < cost(b);
  };

  auto const is_cost_dominated =
      [](n::pareto_set<n::routing::journey> const& pt_journeys,
         n::routing::journey const& odm_journey) {

      };
}

void productivity_domination(std::vector<n::routing::journey>& odm_journeys) {}

void mix(n::pareto_set<n::routing::journey> const& base_journeys,
         std::vector<n::routing::journey>& odm_journeys) {
  cost_domination(base_journeys, odm_journeys);
  productivity_domination(odm_journeys);
  odm_journeys.append_range(base_journeys);
}

}  // namespace motis::odm