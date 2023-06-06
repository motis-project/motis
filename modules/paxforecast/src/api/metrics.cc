#include "motis/paxforecast/api/metrics.h"

#include <cstdint>
#include <vector>

#include "motis/paxforecast/paxforecast.h"
#include "motis/paxforecast/universe_data.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxforecast::api {

msg_ptr metrics(paxforecast& mod, msg_ptr const& msg) {
  auto const req = motis_content(PaxForecastMetricsRequest, msg);
  auto const& data = mod.universe_storage_.get(req->universe());

  message_creator mc;

  auto const metrics_to_fbs = [&](metrics_storage<tick_statistics> const& m) {
    std::vector<std::uint64_t> monitoring_events, group_routes,
        major_delay_group_routes, routing_requests, alternatives_found,
        rerouted_group_routes, removed_group_routes,
        major_delay_group_routes_with_alternatives, total_timing;

    monitoring_events.reserve(m.size());
    group_routes.reserve(m.size());
    major_delay_group_routes.reserve(m.size());
    routing_requests.reserve(m.size());
    alternatives_found.reserve(m.size());
    rerouted_group_routes.reserve(m.size());
    removed_group_routes.reserve(m.size());
    major_delay_group_routes_with_alternatives.reserve(m.size());
    total_timing.reserve(m.size());

    for (auto i = 0UL; i < m.size(); ++i) {
      auto const& entry = m.data_[(m.start_index_ + i) % m.size()];
      monitoring_events.push_back(entry.monitoring_events_);
      group_routes.push_back(entry.group_routes_);
      major_delay_group_routes.push_back(entry.major_delay_group_routes_);
      routing_requests.push_back(entry.routing_requests_);
      alternatives_found.push_back(entry.alternatives_found_);
      rerouted_group_routes.push_back(entry.rerouted_group_routes_);
      removed_group_routes.push_back(entry.removed_group_routes_);
      major_delay_group_routes_with_alternatives.push_back(
          entry.major_delay_group_routes_with_alternatives_);
      total_timing.push_back(entry.t_total_);
    }

    return CreatePaxForecastMetrics(
        mc, m.start_time(), m.size(), mc.CreateVector(monitoring_events),
        mc.CreateVector(group_routes),
        mc.CreateVector(major_delay_group_routes),
        mc.CreateVector(routing_requests), mc.CreateVector(alternatives_found),
        mc.CreateVector(rerouted_group_routes),
        mc.CreateVector(removed_group_routes),
        mc.CreateVector(major_delay_group_routes_with_alternatives),
        mc.CreateVector(total_timing));
  };

  mc.create_and_finish(MsgContent_PaxForecastMetricsResponse,
                       CreatePaxForecastMetricsResponse(
                           mc, metrics_to_fbs(data.metrics_.by_system_time_),
                           metrics_to_fbs(data.metrics_.by_processing_time_))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxforecast::api
