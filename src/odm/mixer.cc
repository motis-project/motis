#include "motis/odm/mixer.h"

#include <filesystem>

#include "boost/json.hpp"

#include "nigiri/logging.h"

#include "motis/metrics_registry.h"
#include "motis/odm/journeys.h"
#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;
namespace nr = nigiri::routing;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace motis::odm {

static constexpr auto kMixerTracing = false;

cost_threshold tag_invoke(boost::json::value_to_tag<cost_threshold>,
                          boost::json::value const& jv) {
  return cost_threshold{static_cast<std::int32_t>(jv.as_array()[0].as_int64()),
                        jv.as_array()[1].as_double()};
}

void tag_invoke(boost::json::value_from_tag,
                boost::json::value& jv,
                cost_threshold const& ct) {
  jv = boost::json::array{ct.threshold_, ct.cost_};
}

std::string label(nr::journey const& j) {
  return std::format("[dep: {}, arr: {}, dur: {}, transfers: {}, odm_time: {}]",
                     j.departure_time(), j.arrival_time(), j.travel_time(),
                     std::uint32_t{j.transfers_}, odm_time(j));
}

void mixer::pareto_dominance(std::vector<nr::journey>& odm_journeys) {

  auto const pareto_dom = [](nr::journey const& a,
                             nr::journey const& b) -> bool {
    auto const odm_time_a = odm_time(a);
    auto const odm_time_b = odm_time(b);
    auto const ret = a.dominates(b) && odm_time_a < odm_time_b;
    if (kMixerTracing) {
      n::log(n::log_lvl::debug, "motis.prima",
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

double tally(std::int64_t const x, std::vector<cost_threshold> const& ct) {
  auto acc = 0.0;
  for (auto i = 0U; i < ct.size() && ct[i].threshold_ < x; ++i) {
    auto const valid_until = i + 1U == ct.size()
                                 ? std::numeric_limits<std::int64_t>::max()
                                 : ct[i + 1U].threshold_;
    acc += static_cast<double>(std::min(x, valid_until) - ct[i].threshold_) *
           ct[i].cost_;
  }
  return acc;
}

double mixer::transfer_cost(nr::journey const& j) const {
  return tally(j.transfers_, transfer_cost_);
}

double mixer::cost(nr::journey const& j) const {
  auto const odm_cost = [&](auto const& l) {
    return tally(std::chrono::abs(l.arr_time_ - l.dep_time_).count(),
                 taxi_cost_);
  };

  auto const odm_cost_first_mile =
      j.legs_.empty() || !is_odm_leg(j.legs_.front(), kOdmTransportModeId)
          ? 0
          : odm_cost(j.legs_.front());

  auto const odm_cost_last_mile =
      j.legs_.size() < 2 || !is_odm_leg(j.legs_.back(), kOdmTransportModeId)
          ? 0
          : odm_cost(j.legs_.back());

  auto const direct_taxi_penalty = is_direct_odm(j) ? direct_taxi_penalty_ : 0;

  return odm_cost_first_mile + pt_time(j).count() + transfer_cost(j) +
         odm_cost_last_mile + direct_taxi_penalty;
};

n::unixtime_t center(nr::journey const& j) {
  return j.departure_time() + j.travel_time() / 2;
}

std::vector<double> mixer::get_threshold(
    std::vector<nr::journey> const& v,
    n::interval<n::unixtime_t> const& intvl,
    double const slope) const {

  if (intvl.from_ >= intvl.to_) {
    return {};
  }

  auto threshold = std::vector(static_cast<size_t>(intvl.size().count()),
                               std::numeric_limits<double>::max());

  for (auto const& j : v) {
    auto const cost_j = cost(j);
    auto const center_j = center(j);
    auto const f_j = [&](n::unixtime_t const t) -> double {
      return slope *
                 static_cast<double>(std::chrono::abs(center_j - t).count()) +
             cost_j;
    };
    for (auto [i, t] = std::tuple{0U, intvl.from_}; t < intvl.to_;
         ++i, t += 1min) {
      threshold[i] = std::min(threshold[i], f_j(t));
    }
  }

  auto const get_next_triangle =
      [&](auto i) -> std::optional<std::tuple<std::uint32_t, std::uint32_t>> {
    auto const get_next_local_minimum =
        [&](auto j) -> std::optional<std::uint32_t> {
      for (; 0 < j && j < threshold.size() - 1; ++j) {
        if (threshold[j - 1] > threshold[j] &&
            threshold[j] < threshold[j + 1]) {
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

    if (threshold[*start] < threshold[*end]) {
      do {
        ++(*start);
      } while (threshold[*start] < threshold[*end]);
    } else if (threshold[*start] > threshold[*end]) {
      do {
        --(*end);
      } while (threshold[*start] > threshold[*end]);
    }

    return std::tuple{*start, *end};
  };

  auto x = 1U;
  while (auto t = get_next_triangle(x)) {
    auto const [start, end] = *t;
    auto const mean =
        std::accumulate(begin(threshold) + start, begin(threshold) + end, 0.0) /
        static_cast<double>(end - start);
    for (auto j = start; j <= end; ++j) {
      threshold[j] = std::min(threshold[j], mean);
    }
    x = end;
  }

  return threshold;
}

void mixer::write_journeys(n::pareto_set<nr::journey> const& pt_journeys,
                           std::vector<nr::journey> const& odm_journeys,
                           std::string_view const stats_path) const {
  auto const journeys_to_csv = [&](auto const& journeys,
                                   auto const& file_name) {
    auto file = std::ofstream{fs::path{stats_path} / file_name};
    file << "departure,center,arrival,travel_time,transfers,odm_time,cost\n";
    for (auto const& j : journeys) {
      file << fmt::format("{},{},{},{},{},{},{}\n", j.departure_time(),
                          center(j), j.arrival_time(), j.travel_time().count(),
                          j.transfers_, odm_time(j).count(), cost(j));
    }
  };
  journeys_to_csv(pt_journeys, "pt_journeys.csv");
  journeys_to_csv(odm_journeys, "odm_journeys.csv");
}

void write_thresholds(std::vector<double> const& pt_threshold,
                      std::vector<double> const& odm_threshold,
                      n::interval<n::unixtime_t> const& intvl,
                      std::string_view const stats_path) {
  auto const threshold_to_csv = [&](auto const& threshold,
                                    auto const& file_name) {
    auto threshold_file = std::ofstream{fs::path{stats_path} / file_name};
    threshold_file << "time,cost\n";
    for (auto const [i, cost] : utl::enumerate(threshold)) {
      threshold_file << fmt::format("{},{}\n", intvl.from_ + n::duration_t{i},
                                    cost);
    }
  };
  threshold_to_csv(pt_threshold, "pt_threshold.csv");
  threshold_to_csv(odm_threshold, "odm_threshold.csv");
}

void add_pt_sort(n::pareto_set<nr::journey> const& pt_journeys,
                 std::vector<nr::journey>& odm_journeys,
                 std::vector<nr::journey> const& ride_share_journeys) {
  for (auto const& j : pt_journeys) {
    odm_journeys.emplace_back(j);
  }
  for (auto const& j : ride_share_journeys) {
    odm_journeys.emplace_back(j);
  }
  utl::sort(odm_journeys, [](auto const& a, auto const& b) {
    return std::tuple{a.departure_time(), a.arrival_time(), a.transfers_} <
           std::tuple{b.departure_time(), b.arrival_time(), b.transfers_};
  });
}

void mixer::mix(n::pareto_set<nr::journey> const& pt_journeys,
                std::vector<nr::journey>& taxi_journeys,
                std::vector<nr::journey> const& ride_share_journeys,
                metrics_registry* metrics,
                std::optional<std::string_view> const stats_path) const {
  pareto_dominance(taxi_journeys);
  auto const pareto_n = taxi_journeys.size();

  if constexpr (kMixerTracing) {
    std::ofstream journeys_in{"journeys.csv"};
    journeys_in << to_csv(get_mixer_input(pt_journeys, taxi_journeys));
  }

  if (stats_path) {
    write_journeys(pt_journeys, taxi_journeys, *stats_path);
  }

  auto const intvl = [&]() {
    auto ret =
        n::interval<n::unixtime_t>{n::unixtime_t::max(), n::unixtime_t::min()};
    for (auto const& j : pt_journeys) {
      ret.from_ = std::min(ret.from_, j.departure_time());
      ret.to_ = std::max(ret.to_, j.arrival_time());
    }
    for (auto const& j : taxi_journeys) {
      ret.from_ = std::min(ret.from_, j.departure_time());
      ret.to_ = std::max(ret.to_, j.arrival_time());
    }
    return ret;
  }();

  auto const threshold_filter = [&](auto const& t) {
    std::erase_if(taxi_journeys, [&](auto const& j) {
      return t[static_cast<size_t>((center(j) - intvl.from_).count())] <
             cost(j);
    });
  };

  auto const pt_threshold = get_threshold(pt_journeys.els_, intvl, pt_slope_);
  threshold_filter(pt_threshold);
  auto const pt_filtered_n = taxi_journeys.size();
  auto const odm_threshold = get_threshold(taxi_journeys, intvl, odm_slope_);
  threshold_filter(odm_threshold);

  if (stats_path) {
    write_thresholds(pt_threshold, odm_threshold, intvl, *stats_path);
  }

  if (metrics != nullptr) {
    metrics->routing_odm_journeys_found_non_dominated_pareto_.Observe(
        static_cast<double>(pareto_n));
    metrics->routing_odm_journeys_found_non_dominated_cost_.Observe(
        static_cast<double>(pt_filtered_n));
    metrics->routing_odm_journeys_found_non_dominated_prod_.Observe(
        static_cast<double>(taxi_journeys.size()));
  }

  add_pt_sort(pt_journeys, taxi_journeys, ride_share_journeys);
}

std::vector<nr::journey> get_mixer_input(
    n::pareto_set<nr::journey> const& pt_journeys,
    std::vector<nr::journey> const& odm_journeys,
    std::vector<nr::journey>& ride_share_journeys) {
  auto ret = odm_journeys;
  add_pt_sort(pt_journeys, ret, ride_share_journeys);
  return ret;
}

mixer get_default_mixer() {
  return mixer{.direct_taxi_penalty_ = 20.0,
               .pt_slope_ = 2.2,
               .odm_slope_ = 2.0,
               .taxi_cost_ = {{0, 20.6}, {1, 4.9}},
               .transfer_cost_ = {{0, 8.0}}};
}

std::ostream& operator<<(std::ostream& o, mixer const& m) {
  return o << boost::json::value_from(m);
}

mixer tag_invoke(boost::json::value_to_tag<mixer>,
                 boost::json::value const& jv) {
  auto m = mixer{};
  m.direct_taxi_penalty_ = jv.at("direct_taxi_penalty").as_double();
  m.pt_slope_ = jv.at("pt_slope").as_double();
  m.odm_slope_ = jv.at("odm_slope").as_double();
  m.taxi_cost_ =
      boost::json::value_to<std::vector<cost_threshold>>(jv.at("taxi_cost"));
  m.transfer_cost_ = boost::json::value_to<std::vector<cost_threshold>>(
      jv.at("transfer_cost"));
  return m;
}

void tag_invoke(boost::json::value_from_tag,
                boost::json::value& jv,
                mixer const& m) {
  jv = boost::json::object{
      {"direct_taxi_penalty_", m.direct_taxi_penalty_},
      {"pt_slope", m.pt_slope_},
      {"odm_slope", m.odm_slope_},
      {"taxi_cost", boost::json::value_from(m.taxi_cost_)},
      {"transfer_cost", boost::json::value_from(m.transfer_cost_)}};
}

}  // namespace motis::odm