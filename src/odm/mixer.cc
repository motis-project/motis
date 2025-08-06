#include "motis/odm/mixer.h"

#include "utl/overloaded.h"

#include "nigiri/logging.h"
#include "nigiri/special_stations.h"

#include "motis/metrics_registry.h"
#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;
namespace nr = nigiri::routing;

static constexpr auto const kMixerTracing = false;

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

std::int32_t mixer::transfer_cost(nr::journey const& j) const {
  return tally(j.transfers_, transfer_cost_);
}

std::int32_t distance(nr::journey const& a, nr::journey const& b) {
  auto const overtakes = [](auto const& x, auto const& y) {
    return x.departure_time() > y.departure_time() &&
           x.arrival_time() < y.arrival_time();
  };

  return overtakes(a, b) || overtakes(b, a)
             ? 0
             : std::max(
                   std::chrono::abs(a.departure_time() - b.departure_time()),
                   std::chrono::abs(a.arrival_time() - b.arrival_time()))
                   .count();
}

std::string label(nr::journey const& j) {
  return std::format("[dep: {}, arr: {}, dur: {}, transfers: {}, odm_time: {}]",
                     j.departure_time(), j.arrival_time(), j.travel_time(),
                     std::uint32_t{j.transfers_}, odm_time(j));
}

double mixer::cost(nr::journey const& j) const {

  auto const leg_cost = [&](nr::journey::leg const& leg) {
    return std::visit(
        utl::overloaded{
            [](nr::journey::run_enter_exit const&) { return std::int32_t{0}; },
            [&](n::footpath const& fp) {
              return tally(fp.duration().count(), walk_cost_);
            },
            [&](nr::offset const& o) {
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

  auto const pt_time = [](nr::journey const& j) {
    auto const leg_duration = [](nr::journey::leg const& l) {
      return std::visit(
          utl::overloaded{[](nr::journey::run_enter_exit const&) {
                            return n::duration_t{0};
                          },
                          [](n::footpath const& fp) { return fp.duration(); },
                          [](nr::offset const& o) { return o.duration(); }},
          l.uses_);
    };
    return j.travel_time() - leg_duration(j.legs_.front()) -
           ((j.legs_.size() > 1) ? leg_duration(j.legs_.back())
                                 : n::duration_t{0});
  };

  return leg_cost(j.legs_.front()) +
         (j.legs_.size() > 1 ? leg_cost(j.legs_.back()) : 0) +
         pt_time(j).count() + transfer_cost(j);
};

bool mixer::cost_dominates(nr::journey const& a, nr::journey const& b) const {
  auto const cost_a = cost(a);
  auto const cost_b =
      cost(b) + (is_pure_pt(a) && is_direct_odm(b) ? direct_taxi_penalty_ : 0);
  auto const time_ratio = static_cast<double>(a.travel_time().count()) /
                          static_cast<double>(b.travel_time().count());
  auto const dist = std::max(distance(a, b), min_distance_);
  auto const alpha_term =
      cost_alpha_ * std::min(time_ratio, 3.0) * std::pow(dist, exp_distance_);
  auto const ret = dist < max_distance_ && cost_a + alpha_term < cost_b;
  if (kMixerTracing) {
    n::log(n::log_lvl::debug, "motis.odm",
           "{} cost-dominates {}, ratio: {:.2f}, dist: {}, {:.2f} + {:.2f} < "
           "{:.2f} --> {}",
           label(a), label(b), time_ratio, dist, cost_a, alpha_term, cost_b,
           ret ? "true" : "false");
  }
  return ret;
}

void mixer::cost_dominance(
    nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
    std::vector<nigiri::routing::journey>& odm_journeys) const {
  auto const is_dominated = [&](nr::journey const& odm_journey) {
    auto const dominates = [&](nr::journey const& pt_journey) -> bool {
      return cost_dominates(pt_journey, odm_journey);
    };

    return utl::any_of(pt_journeys, dominates);
  };

  std::erase_if(odm_journeys, is_dominated);
}

void establish_dominance(
    std::vector<nr::journey>& journeys,
    std::function<bool(nr::journey const&, nr::journey const&)> const&
        dominates) {
  for (auto b = begin(journeys); b != end(journeys);) {
    auto is_dominated = false;
    for (auto a = begin(journeys); a != end(journeys); ++a) {
      if (a != b && dominates(*a, *b)) {
        is_dominated = true;
        break;
      }
    }
    if (is_dominated) {
      b = journeys.erase(b);
    } else {
      ++b;
    }
  }
}

void mixer::pareto_dominance(
    std::vector<nigiri::routing::journey>& odm_journeys) {

  auto const pareto_dom = [](nr::journey const& a,
                             nr::journey const& b) -> bool {
    auto const odm_time_a = odm_time(a);
    auto const odm_time_b = odm_time(b);
    auto const ret = a.dominates(b) && odm_time_a < odm_time_b;
    if (kMixerTracing) {
      n::log(n::log_lvl::debug, "motis.odm",
             "{} pareto-dominates {}, odm_time: {} < {} --> {}", label(a),
             label(b), odm_time_a, odm_time_b, ret ? "true" : "false");
    }
    return ret;
  };

  establish_dominance(odm_journeys, pareto_dom);
}

void mixer::productivity_dominance(
    std::vector<nr::journey>& odm_journeys) const {

  auto const prod_cost = [&](nr::journey const& j) {
    return static_cast<double>(j.travel_time().count() + transfer_cost(j));
  };

  auto const prod_dom = [&](nr::journey const& a, nr::journey const& b) {
    auto const cost_a = prod_cost(a);
    auto const cost_b = prod_cost(b);
    auto const odm_time_a = static_cast<double>(odm_time(a).count());
    auto const odm_time_b = static_cast<double>(odm_time(b).count());
    auto const dist = std::max(distance(a, b), min_distance_);
    auto const alpha_term = prod_alpha_ * std::pow(dist, exp_distance_);
    auto const prod_a = cost_b / odm_time_a;
    auto const prod_b = (cost_a + alpha_term) / odm_time_b;
    auto const ret = dist < max_distance_ && prod_a > prod_b;
    if (kMixerTracing) {
      n::log(n::log_lvl::debug, "motis.odm",
             "{} prod-dominates {}, dist: {}, {} > {} --> {}", label(a),
             label(b), dist, prod_a, prod_b, ret ? "true" : "false");
    }
    return ret;
  };

  establish_dominance(odm_journeys, prod_dom);
}

void mixer::mix(n::pareto_set<nr::journey> const& pt_journeys,
                std::vector<nr::journey>& odm_journeys,
                metrics_registry* metrics) const {
  pareto_dominance(odm_journeys);
  auto const pareto_n = odm_journeys.size();
  cost_dominance(pt_journeys, odm_journeys);
  auto const cost_n = odm_journeys.size();
  productivity_dominance(odm_journeys);

  if (metrics != nullptr) {
    metrics->routing_odm_journeys_found_non_dominated_pareto_.Observe(
        static_cast<double>(pareto_n));
    metrics->routing_odm_journeys_found_non_dominated_cost_.Observe(
        static_cast<double>(cost_n));
    metrics->routing_odm_journeys_found_non_dominated_prod_.Observe(
        static_cast<double>(odm_journeys.size()));
  }

  for (auto const& j : pt_journeys) {
    odm_journeys.emplace_back(j);
  }
  utl::sort(odm_journeys, [](auto const& a, auto const& b) {
    return a.departure_time() < b.departure_time();
  });
}

mixer get_default_mixer() {
  return mixer{.cost_alpha_ = 1.3,
               .prod_alpha_ = 0.4,
               .direct_taxi_penalty_ = 220,
               .min_distance_ = 15,
               .max_distance_ = 90,
               .exp_distance_ = 1.045,
               .walk_cost_ = {{0, 1}, {15, 10}},
               .taxi_cost_ = {{0, 35}, {1, 12}},
               .transfer_cost_ = {{0, 10}}};
}

}  // namespace motis::odm