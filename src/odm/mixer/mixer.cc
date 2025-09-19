#include "motis/odm/mixer/mixer.h"

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

std::string label(nr::journey const& j) {
  return std::format("[dep: {}, arr: {}, dur: {}, transfers: {}, odm_time: {}]",
                     j.departure_time(), j.arrival_time(), j.travel_time(),
                     std::uint32_t{j.transfers_}, odm_time(j));
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

  for (auto b = begin(odm_journeys); b != end(odm_journeys);) {
    auto is_dominated = false;
    for (auto a = begin(odm_journeys); a != end(odm_journeys); ++a) {
      if (a != b && pareto_dom(*a, *b)) {
        is_dominated = true;
        break;
      }
    }
    if (is_dominated) {
      b = odm_journeys.erase(b);
    } else {
      ++b;
    }
  }
}

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
              } else if (o.transport_mode_id_ == kWalkTransportModeId) {
                return tally(o.duration().count(), walk_cost_);
              }
              utl::verify(o.transport_mode_id_ == kOdmTransportModeId ||
                              o.transport_mode_id_ == kWalkTransportModeId,
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

void mixer::cost_dominance(
    nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
    std::vector<nigiri::routing::journey>& odm_journeys) const {

  auto const center = [](nr::journey const& j) -> n::unixtime_t {
    return j.departure_time() + j.travel_time() / 2;
  };

  auto const intvl = [&]() {
    auto ret =
        n::interval<n::unixtime_t>{n::unixtime_t::max(), n::unixtime_t::min()};
    for (auto const& j : pt_journeys) {
      ret.from_ = std::min(ret.from_, j.departure_time());
      ret.to_ = std::max(ret.to_, j.arrival_time());
    }
    for (auto const& j : odm_journeys) {
      ret.from_ = std::min(ret.from_, j.departure_time());
      ret.to_ = std::max(ret.to_, j.arrival_time());
    }
    return ret;
  }();

  auto cost_threshold = [&]() {
    auto ret = std::vector<double>(
        intvl.size().count(), std::numeric_limits<double>::max());
    for (auto const& j : pt_journeys) {
      auto const cost_j = cost(j);
      auto const center_j = center(j);
      auto const f_j = [&](n::unixtime_t const t) -> double {
        return cost_j * (1.0 + static_cast<double>(
                                   std::chrono::abs(center_j - t).count()) /
                                   static_cast<double>(max_distance_));
      };
      for (auto const [i, t] : utl::enumerate(intvl)) {
        ret[i] = std::min(ret[i], f_j(t));
      }
    }
    return ret;
  }();

  auto const get_next_triangle =
      [&](auto i) -> std::optional<std::tuple<std::uint32_t, std::uint32_t>> {
    auto const get_next_local_minimum =
        [&](auto j) -> std::optional<std::uint32_t> {
      for (; 0 < j && j < cost_threshold.size() - 1; ++j) {
        if (cost_threshold[j - 1] > cost_threshold[j] &&
            cost_threshold[j] < cost_threshold[j + 1]) {
          return j;
        }
      }
      return std::nullopt;
    };

    auto start = get_next_local_minimum(i);
    if (!start) {
      return std::nullopt;
    }
    auto end = get_next_local_minimum(*start + 1);
    if (!end) {
      return std::nullopt;
    }

    if (cost_threshold[*start] < cost_threshold[*end]) {
      do {
        ++(*start);
      } while (cost_threshold[*start] < cost_threshold[*end]);
    } else if (cost_threshold[*start] > cost_threshold[*end]) {
      do {
        --(*end);
      } while (cost_threshold[*start] > cost_threshold[*end]);
    }

    return std::tuple{*start, *end};
  };

  auto x = 1U;
  while (auto t = get_next_triangle(x)) {
    auto const [start, end] = *t;
    auto const mean = std::accumulate(begin(cost_threshold) + start,
                                      begin(cost_threshold) + end, 0.0) /
                      static_cast<double>(end - start);
    for (auto j = start; j <= end; ++j) {
      cost_threshold[j] = std::min(cost_threshold[j], mean);
    }
    x = end;
  }

  if constexpr (true) {
    auto cost_threshold_file = std::ofstream{"cost_threshold.csv"};
    cost_threshold_file << "time,cost\n";
    for (auto const [i, cost] : utl::enumerate(cost_threshold)) {
      cost_threshold_file << fmt::format("{},{}\n",
                                         intvl.from_ + n::duration_t{i}, cost);
    }
    auto const to_csv = [&](auto const& journeys, auto const& file_name) {
      auto file = std::ofstream{file_name};
      file << "departure,center,arrival,travel_time,transfers,odm_time,cost\n";
      for (auto const& j : journeys) {
        file << fmt::format("{},{},{},{},{},{},{}\n", j.departure_time(),
                                     center(j), j.arrival_time(), j.travel_time(), j.transfers_, odm_time(j), cost(j));
      }
    };
    to_csv(pt_journeys, "pt_journeys.csv");
    to_csv(odm_journeys, "odm_journeys.csv");
  }

  std::erase_if(odm_journeys, [&](auto const& j) {
    return cost_threshold[(center(j) - intvl.from_).count()] <= cost(j);
  });
}


void add_pt_sort(n::pareto_set<nr::journey> const& pt_journeys,
                 std::vector<nr::journey>& odm_journeys) {
  for (auto const& j : pt_journeys) {
    odm_journeys.emplace_back(j);
  }
  utl::sort(odm_journeys, [](auto const& a, auto const& b) {
    return std::tuple{a.departure_time(), a.arrival_time(), a.transfers_} <
           std::tuple{b.departure_time(), b.arrival_time(), b.transfers_};
  });
}

void mixer::mix(n::pareto_set<nr::journey> const& pt_journeys,
                std::vector<nr::journey>& odm_journeys,
                metrics_registry* metrics) const {
  pareto_dominance(odm_journeys);
  auto const pareto_n = odm_journeys.size();
  cost_dominance(pt_journeys, odm_journeys);
  auto const cost_n = odm_journeys.size();

  if (metrics != nullptr) {
    metrics->routing_odm_journeys_found_non_dominated_pareto_.Observe(
        static_cast<double>(pareto_n));
    metrics->routing_odm_journeys_found_non_dominated_cost_.Observe(
        static_cast<double>(cost_n));
    metrics->routing_odm_journeys_found_non_dominated_prod_.Observe(
        static_cast<double>(odm_journeys.size()));
  }

  add_pt_sort(pt_journeys, odm_journeys);
}

std::vector<nr::journey> get_mixer_input(
    n::pareto_set<nr::journey> const& pt_journeys,
    std::vector<nr::journey> const& odm_journeys) {
  auto ret = odm_journeys;
  add_pt_sort(pt_journeys, ret);
  return ret;
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