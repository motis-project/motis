#include "motis/odm/mixer.h"

#include "utl/overloaded.h"

#include "nigiri/special_stations.h"

#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;
using n::routing::journey;

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

std::int32_t mixer::transfer_cost(journey const& j) const {
  return tally(j.transfers_, transfer_cost_);
}

std::int32_t distance(journey const& a, journey const& b) {
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

std::string journey_label(n::routing::journey const& j) {
  return std::format("[dep: {}, arr: {}, dur: {}, transfers: {}]",
                     j.departure_time(), j.arrival_time(), j.travel_time(),
                     std::uint32_t{j.transfers_});
}

void mixer::cost_domination(auto& journeys,
                            std::vector<journey>& odm_journeys) const {
  auto const leg_cost = [&](journey::leg const& leg) {
    return std::visit(
        utl::overloaded{
            [](journey::run_enter_exit const&) { return std::int32_t{0}; },
            [&](n::footpath const& fp) {
              return tally(fp.duration().count(), walk_cost_);
            },
            [&](n::routing::offset const& o) {
              if (o.transport_mode_id_ == kOdmTransportModeId) {
                return tally(o.duration().count(), taxi_cost_);
              } else if (o.transport_mode_id_ == kWalk) {
                return tally(o.duration().count(), walk_cost_);
              }
              utl::verify(o.transport_mode_id_ == kOdmTransportModeId ||
                              o.transport_mode_id_ == kWalk,
                          "unknown transport mode");
              return std::int32_t{0};
            }},
        leg.uses_);
  };

  auto const pt_time = [](journey const& j) {
    auto const leg_duration = [](journey::leg const& l) {
      return std::visit(
          utl::overloaded{
              [](journey::run_enter_exit const& ree [[maybe_unused]]) {
                return n::duration_t{0};
              },
              [](n::footpath const& fp) { return fp.duration(); },
              [](n::routing::offset const& o) { return o.duration(); }},
          l.uses_);
    };
    return j.travel_time() - leg_duration(j.legs_.front()) -
           ((j.legs_.size() > 1) ? leg_duration(j.legs_.back())
                                 : n::duration_t{0});
  };

  auto const cost = [&](journey const& j) {
    return (leg_cost(j.legs_.front()) +
            (j.legs_.size() > 1 ? leg_cost(j.legs_.back()) : 0) +
            pt_time(j).count() + transfer_cost(j));
  };

  auto const dominates = [&](n::routing::journey const& journey,
                             n::routing::journey const& odm_journey) -> bool {
    auto const cost_pt = cost(journey);
    auto const time_ratio =
        static_cast<double>(journey.travel_time().count()) /
        static_cast<double>(odm_journey.travel_time().count());
    auto const dist = distance(journey, odm_journey);
    auto const alpha_term = alpha_ * time_ratio * dist;
    auto const cost_odm = cost(odm_journey);
    auto const ret = cost_pt + alpha_term < cost_odm;
    if (kMixerTracing && ret) {
      fmt::println(
          "{} cost-dominates {}\ntime_ratio: {} / {} = {}, distance: {}, "
          "alpha_term: {} * {} * {} = {}, {} + {} < {}",
          journey_label(journey), journey_label(odm_journey),
          journey.travel_time(), odm_journey.travel_time(), time_ratio, dist,
          alpha_, time_ratio, dist, alpha_term, cost_pt, alpha_term, cost_odm);
    }
    return ret;
  };

  for (auto odm_journey = begin(odm_journeys);
       odm_journey != end(odm_journeys);) {
    auto is_dominated = false;
    for (auto const& journey : journeys) {
      if (dominates(journey, *odm_journey)) {
        is_dominated = true;
        break;
      }
    }
    if (is_dominated) {
      odm_journeys.erase(odm_journey);
    } else {
      ++odm_journey;
    }
  }
}

void mixer::pareto_domination(
    std::vector<n::routing::journey>& odm_journeys) const {

  auto const is_dominated = [&](n::routing::journey const& b) {
    auto const dominates = [&](n::routing::journey const& a) {
      auto const odm_time_a = odm_time(a);
      auto const odm_time_b = odm_time(b);
      auto const ret = a.dominates(b) && odm_time_a < odm_time_b;
      if (kMixerTracing && ret) {
        fmt::println("{} pareto-dominates {}\nodm_time: {} < {}",
                     journey_label(a), journey_label(b), odm_time_a,
                     odm_time_b);
      }
      return ret;
    };

    return utl::any_of(odm_journeys, dominates);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void mixer::productivity_domination(std::vector<journey>& odm_journeys) const {
  auto const cost = [&](auto const& j) -> double {
    return j.travel_time().count() + transfer_cost(j);
  };

  auto const is_dominated = [&](journey const& b) {
    auto const dominates = [&](journey const& a) {
      auto const cost_a = cost(a);
      auto const cost_b = cost(b);
      auto const taxi_time_a = static_cast<double>(odm_time(a).count());
      auto const taxi_time_b = static_cast<double>(odm_time(b).count());
      auto const prod_a = cost_b / taxi_time_a;
      auto const dist = distance(a, b);
      auto const prod_b = (cost(a) + beta_ * dist) / taxi_time_b;
      auto const ret = prod_a > prod_b;
      if (kMixerTracing && ret) {
        fmt::println(
            "{} prod-dominates {}\nprod_a = cost(b) / taxi_time(a) = {} / {} = "
            "{}, prod_b = (cost(a) + beta * distance(a,b)) / taxi_time(b) = "
            "({} + {} * {}) / {} = {}, "
            "prod_a > prod_b <=> {} > {}",
            journey_label(a), journey_label(b), cost_b, taxi_time_a, prod_a,
            cost_a, beta_, dist, taxi_time_b, prod_b, prod_a, prod_b);
      }
      return ret;
    };
    return utl::any_of(odm_journeys, dominates);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void mixer::mix(n::pareto_set<journey> const& pt_journeys,
                std::vector<journey>& odm_journeys) const {
  cost_domination(pt_journeys, odm_journeys);
  pareto_domination(odm_journeys);
  // productivity_domination(odm_journeys);
  for (auto const& j : pt_journeys) {
    odm_journeys.emplace_back(j);
  }
  utl::sort(odm_journeys, [](auto const& a, auto const& b) {
    return a.departure_time() < b.departure_time();
  });
}

}  // namespace motis::odm