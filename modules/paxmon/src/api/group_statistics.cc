#include "motis/paxmon/api/group_statistics.h"

#include <cmath>
#include <algorithm>

#include "utl/enumerate.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

namespace {

struct histogram {
  histogram(int const lowest_allowed, int const highest_allowed)
      : offset_{-lowest_allowed},
        lowest_allowed_{lowest_allowed},
        highest_allowed_{highest_allowed},
        min_value_{highest_allowed},
        max_value_{lowest_allowed_} {
    counts_.resize(highest_allowed - lowest_allowed + 1);
  }

  void add(int value, int const count = 1) {
    value = std::clamp(value, lowest_allowed_, highest_allowed_);
    counts_[value + offset_] += count;
    total_count_ += count;
    if (value < min_value_) {
      min_value_ = value;
    }
    if (value > max_value_) {
      max_value_ = value;
    }
  }

  void finish() {
    if (min_value_ > -offset_) {
      auto const empty_beginning = min_value_ + offset_;
      counts_.erase(counts_.begin(),
                    std::next(counts_.begin(), empty_beginning));
      offset_ -= empty_beginning;
    }

    if ((max_value_ + offset_) < counts_.size() - 1) {
      auto const empty_end = counts_.size() - (max_value_ + offset_) - 1;
      counts_.resize(counts_.size() - empty_end);
    }

    avg_value_ = 0;
    median_value_ = 0;
    max_count_ = 0;
    auto count_sum = 0.;
    auto const half_total_count = static_cast<double>(total_count_) / 2.0;
    auto calc_median = true;
    for (auto const [index, count] : utl::enumerate(counts_)) {
      auto const value = static_cast<int>(index) - offset_;
      avg_value_ += value * (static_cast<double>(count) /
                             static_cast<double>(total_count_));
      if (count > max_count_) {
        max_count_ = count;
      }
      if (calc_median) {
        count_sum += count;
        if (count_sum >= half_total_count) {
          median_value_ = value;
          calc_median = false;
        }
      }
    }
  }

  std::vector<unsigned> counts_;
  int offset_;
  int lowest_allowed_;
  int highest_allowed_;
  int min_value_;
  int max_value_;
  unsigned max_count_{};
  unsigned total_count_{};
  double avg_value_{};
  double median_value_{};
};

auto const constexpr LOWEST_ALLOWED_DELAY = -1440;
auto const constexpr HIGHEST_ALLOWED_DELAY = 1440;
auto const constexpr HIGHEST_ROUTES_PER_GROUP = 1000;
auto const constexpr HIGHEST_REROUTES_PER_GROUP = 1000;

}  // namespace

msg_ptr group_statistics(paxmon_data& data, motis::module::msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGroupStatisticsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& uv = uv_access.uv_;
  auto const& pgc = uv.passenger_groups_;
  auto const count_passengers = req->count_passengers();

  auto h_min_est_delay = histogram{LOWEST_ALLOWED_DELAY, HIGHEST_ALLOWED_DELAY};
  auto h_max_est_delay = histogram{LOWEST_ALLOWED_DELAY, HIGHEST_ALLOWED_DELAY};
  auto h_expected_est_delay =
      histogram{LOWEST_ALLOWED_DELAY, HIGHEST_ALLOWED_DELAY};
  auto h_routes_per_group = histogram{0, HIGHEST_ROUTES_PER_GROUP};
  auto h_active_routes_per_group = histogram{0, HIGHEST_ROUTES_PER_GROUP};
  auto h_reroutes_per_group = histogram{0, HIGHEST_REROUTES_PER_GROUP};
  auto h_group_route_probabilities = histogram{0, 100};

  auto total_group_route_count = 0U;
  auto active_group_route_count = 0U;
  auto unreachable_dest_group_count = 0U;
  auto total_pax_count = 0ULL;

  for (auto const& pg : pgc) {
    auto const pgi = pg->id_;
    auto const routes = pgc.routes(pgi);
    auto const reroute_log = pgc.reroute_log_entries(pgi);
    h_routes_per_group.add(routes.size());
    h_reroutes_per_group.add(reroute_log.size());
    total_pax_count += pg->passengers_;

    auto min_estimated_delay = HIGHEST_ALLOWED_DELAY;
    auto max_estimated_delay = LOWEST_ALLOWED_DELAY;
    auto expected_estimated_delay = 0.F;
    auto active_routes = 0U;
    auto has_unreachable_dest_routes = false;
    for (auto const& gr : routes) {
      if (gr.probability_ == 0) {
        continue;
      }
      ++active_routes;
      if (gr.estimated_delay_ < min_estimated_delay) {
        min_estimated_delay = gr.estimated_delay_;
      }
      if (gr.estimated_delay_ > max_estimated_delay) {
        max_estimated_delay = gr.estimated_delay_;
      }
      expected_estimated_delay += gr.probability_ * gr.estimated_delay_;
      h_group_route_probabilities.add(
          static_cast<int>(std::round(gr.probability_ * 100)));
      if (gr.destination_unreachable_) {
        has_unreachable_dest_routes = true;
      }
    }
    h_active_routes_per_group.add(active_routes);
    total_group_route_count += routes.size();
    active_group_route_count += active_routes;
    if (has_unreachable_dest_routes) {
      ++unreachable_dest_group_count;
    }
    if (active_routes == 0) {
      continue;
    }

    auto const count = count_passengers ? pg->passengers_ : 1;

    h_min_est_delay.add(min_estimated_delay, count);
    h_max_est_delay.add(max_estimated_delay, count);
    h_expected_est_delay.add(
        static_cast<unsigned>(std::round(expected_estimated_delay)), count);
  }

  message_creator mc;

  auto const histogram_to_fbs = [&](histogram& h) {
    h.finish();
    return CreatePaxMonHistogram(mc, h.min_value_, h.max_value_, h.avg_value_,
                                 h.median_value_, h.max_count_, h.total_count_,
                                 mc.CreateVector(h.counts_));
  };

  mc.create_and_finish(
      MsgContent_PaxMonGroupStatisticsResponse,
      CreatePaxMonGroupStatisticsResponse(
          mc, uv.passenger_groups_.size(), total_group_route_count,
          active_group_route_count, unreachable_dest_group_count,
          total_pax_count, histogram_to_fbs(h_min_est_delay),
          histogram_to_fbs(h_max_est_delay),
          histogram_to_fbs(h_expected_est_delay),
          histogram_to_fbs(h_routes_per_group),
          histogram_to_fbs(h_active_routes_per_group),
          histogram_to_fbs(h_reroutes_per_group),
          histogram_to_fbs(h_group_route_probabilities))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
