#include "motis/paxforecast/update_tracked_groups.h"

#include <iostream>
#include <vector>

#include "utl/verify.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/temp_passenger_group.h"

using namespace motis::paxmon;
using namespace motis::module;

namespace motis::paxforecast {

auto const constexpr REROUTE_BATCH_SIZE = 5'000;

inline reroute_reason_t to_reroute_reason(monitoring_event_type const met) {
  switch (met) {
    case monitoring_event_type::BROKEN_TRANSFER:
      return reroute_reason_t::BROKEN_TRANSFER;
    case monitoring_event_type::MAJOR_DELAY_EXPECTED:
      return reroute_reason_t::MAJOR_DELAY_EXPECTED;
    case monitoring_event_type::NO_PROBLEM:
      return reroute_reason_t::UPDATE_FORECAST;
  }
  throw utl::fail("to_reroute_reason: unhandled monitoring_event_type");
}

void update_tracked_groups(
    schedule const& sched, universe& uv, simulation_result const& sim_result,
    std::map<passenger_group_with_route, monitoring_event_type> const&
        pgwr_event_types,
    std::map<passenger_group_with_route,
             std::optional<broken_transfer_info>> const& broken_transfer_infos,
    mcd::hash_map<passenger_group_with_route,
                  passenger_localization const*> const& pgwr_localizations,
    tick_statistics& tick_stats,
    reroute_reason_t const default_reroute_reason) {
  using namespace flatbuffers;

  message_creator mc;
  auto reroutes = std::vector<Offset<PaxMonRerouteGroup>>{};
  auto reroute_count = 0;

  auto const send_reroutes = [&]() {
    if (reroutes.empty()) {
      return;
    }
    mc.create_and_finish(
        MsgContent_PaxMonRerouteGroupsRequest,
        CreatePaxMonRerouteGroupsRequest(mc, uv.id_, mc.CreateVector(reroutes))
            .Union(),
        "/paxmon/reroute_groups");
    auto const msg = make_msg(mc);
    motis_call(msg)->val();
    reroutes.clear();
    mc.Clear();
  };

  for (auto const& [pgwr, result] : sim_result.group_route_results_) {

    auto reroute_reason = default_reroute_reason;
    if (auto const it = pgwr_event_types.find(pgwr);
        it != end(pgwr_event_types)) {
      reroute_reason = to_reroute_reason(it->second);
      if (reroute_reason == reroute_reason_t::UPDATE_FORECAST) {
        std::cout << "update_tracked_groups: UPDATE_FORECAST NYI\n";
        continue;
      }
    }

    auto& gr = uv.passenger_groups_.route(pgwr);

    if (result.alternative_probabilities_.empty()) {
      // keep existing group (only reachable part)
      reroute_reason = reroute_reason_t::DESTINATION_UNREACHABLE;
      gr.destination_unreachable_ = true;
    }

    auto const old_journey =
        uv.passenger_groups_.journey(gr.compact_journey_index_);
    auto const journey_prefix =
        get_prefix(sched, old_journey, *result.localization_);

    // major delay groups have already been removed

    // add alternatives
    auto new_routes = std::vector<Offset<PaxMonGroupRoute>>{};
    for (auto const& [alt, prob] : result.alternative_probabilities_) {
      if (prob == 0.0) {
        continue;
      }

      compact_journey new_journey;
      try {
        new_journey =
            merge_journeys(sched, journey_prefix, alt->compact_journey_);
      } catch (std::runtime_error const& e) {
        std::cout << "\noriginal planned journey:\n";
        print_compact_journey(sched, old_journey);
        std::cout << "\nlocalization: in_trip="
                  << result.localization_->in_trip()
                  << ", first_station=" << result.localization_->first_station_
                  << ", station="
                  << result.localization_->at_station_->name_.str()
                  << ", schedule_arrival_time="
                  << format_time(result.localization_->schedule_arrival_time_)
                  << ", current_arrival_time="
                  << format_time(result.localization_->current_arrival_time_)
                  << "\n";
        if (result.localization_->in_trip()) {
          print_trip(sched, result.localization_->in_trip_);
        }
        throw e;
      }

      auto const tgr = temp_group_route{
          std::nullopt /* index */,
          prob,
          new_journey,
          gr.planned_arrival_time_,
          0 /* estimated delay - updated by reroute groups api */,
          route_source_flags::FORECAST,
          false /* planned */};

      new_routes.emplace_back(to_fbs(sched, mc, tgr));
    }

    auto const bti_it = broken_transfer_infos.find(pgwr);
    auto fbs_localization =
        std::vector<flatbuffers::Offset<PaxMonLocalizationWrapper>>{};
    if (auto const it = pgwr_localizations.find(pgwr);
        it != end(pgwr_localizations)) {
      fbs_localization.emplace_back(
          to_fbs_localization_wrapper(sched, mc, *it->second));
    }
    reroutes.emplace_back(CreatePaxMonRerouteGroup(
        mc, pgwr.pg_, pgwr.route_, mc.CreateVector(new_routes),
        static_cast<PaxMonRerouteReason>(reroute_reason),
        broken_transfer_info_to_fbs(mc, sched,
                                    bti_it != end(broken_transfer_infos)
                                        ? bti_it->second
                                        : std::nullopt),
        false, mc.CreateVector(fbs_localization)));
    ++reroute_count;

    if (reroutes.size() >= REROUTE_BATCH_SIZE) {
      send_reroutes();
    }
  }

  send_reroutes();

  tick_stats.rerouted_group_routes_ += reroute_count;
}

}  // namespace motis::paxforecast
