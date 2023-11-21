#pragma once

#include <cstdint>
#include <algorithm>
#include <vector>

#include "utl/zip.h"

#include "motis/core/schedule/schedule.h"

#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/message.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/reroute_log_entry.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/affected_route_info.h"
#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast {

struct simulation_options {
  float probability_threshold_{};
  float uninformed_pax_{};
};

struct update_groups_context {
  motis::module::message_creator mc_;
  std::vector<flatbuffers::Offset<motis::paxmon::PaxMonRerouteGroup>> reroutes_;
};

template <typename PassengerBehavior>
inline void simulate_behavior_for_alternatives(PassengerBehavior& pb,
                                               std::vector<alternative>& alts) {
  auto const allocation = pb.pick_routes(alts);
  for (auto const& [alt, probability] : utl::zip(alts, allocation)) {
    alt.pick_probability_ = probability;
  }
}

template <typename PassengerBehavior>
inline void simulate_behavior_for_alternatives(PassengerBehavior& pb,
                                               alternatives_set& alts_set) {
  motis_parallel_for(alts_set.requests_, [&](auto& req) {
    simulate_behavior_for_alternatives(pb, req.alternatives_);
  });
}

void simulate_behavior_for_route(
    schedule const& sched, motis::paxmon::universe& uv,
    update_groups_context& ug_ctx, simulation_options const& options,
    affected_route_info const& ar, std::vector<alternative> const& alts_now,
    std::vector<alternative> const& alts_broken,
    motis::paxmon::reroute_reason_t const default_reroute_reason);

}  // namespace motis::paxforecast
