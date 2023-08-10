#include "motis/paxmon/api/metrics.h"

#include <cstdint>
#include <vector>

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr metrics(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonMetricsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& uv = uv_access.uv_;

  message_creator mc;

  auto const metrics_to_fbs = [&](metrics_storage<tick_statistics> const& m) {
    std::vector<std::uint64_t> affected_group_routes, ok_group_routes,
        broken_group_routes, major_delay_group_routes, total_timing;

    affected_group_routes.reserve(m.size());
    ok_group_routes.reserve(m.size());
    broken_group_routes.reserve(m.size());
    major_delay_group_routes.reserve(m.size());
    total_timing.reserve(m.size());

    for (auto i = 0UL; i < m.size(); ++i) {
      auto const& entry = m.data_[(m.start_index_ + i) % m.size()];
      affected_group_routes.push_back(entry.affected_group_routes_);
      ok_group_routes.push_back(entry.ok_group_routes_);
      broken_group_routes.push_back(entry.broken_group_routes_);
      major_delay_group_routes.push_back(entry.major_delay_group_routes_);
      total_timing.push_back(entry.t_rt_updates_applied_total_);
    }

    return CreatePaxMonMetrics(
        mc, m.start_time(), m.size(), mc.CreateVector(affected_group_routes),
        mc.CreateVector(ok_group_routes), mc.CreateVector(broken_group_routes),
        mc.CreateVector(major_delay_group_routes),
        mc.CreateVector(total_timing));
  };

  mc.create_and_finish(MsgContent_PaxMonMetricsResponse,
                       CreatePaxMonMetricsResponse(
                           mc, metrics_to_fbs(uv.metrics_.by_system_time_),
                           metrics_to_fbs(uv.metrics_.by_processing_time_))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
