#include "motis/paxforecast/messages.h"

#include <cassert>
#include <algorithm>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace flatbuffers;
using namespace motis::paxmon;

namespace motis::paxforecast {

Offset<PassengerForecastResult> to_fbs(schedule const& sched,
                                       FlatBufferBuilder& fbb,
                                       simulation_result const& res,
                                       graph const& g) {
  auto const trip_with_edges_to_fbs =
      [&](trip const* trp, std::vector<motis::paxmon::edge*> const& edges) {
        std::vector<Offset<EdgeOverCapacity>> fb_edges;
        for (auto const e : edges) {
          if (e->type_ != edge_type::TRIP) {
            continue;
          }
          auto const from = e->from(g);
          auto const to = e->to(g);
          fb_edges.emplace_back(CreateEdgeOverCapacity(
              fbb, e->passengers(), e->capacity(),
              res.additional_passengers_.at(e),
              to_fbs(fbb, from->get_station(sched)),
              to_fbs(fbb, to->get_station(sched)),
              motis_to_unixtime(sched, from->schedule_time()),
              motis_to_unixtime(sched, to->schedule_time())));
        }
        return CreateTripOverCapacity(fbb, to_fbs(sched, fbb, trp),
                                      fbb.CreateVector(fb_edges));
      };

  return CreatePassengerForecastResult(
      fbb, res.is_over_capacity(), res.edge_count_over_capacity(),
      res.total_passengers_over_capacity(),
      fbb.CreateVector(utl::to_vec(
          res.trips_over_capacity_with_edges(), [&](auto const& kv) {
            return trip_with_edges_to_fbs(kv.first, kv.second);
          })));
}

Offset<CompactJourney> get_forecast_journey(
    schedule const& sched, FlatBufferBuilder& fbb,
    combined_passenger_group const& cpg,
    std::vector<std::uint16_t> const& allocations) {
  // TODO(pablo): temp solution
  assert(allocations.size() == cpg.alternatives_.size());
  if (cpg.alternatives_.empty()) {
    return to_fbs(sched, fbb, compact_journey{{}});
  }
  auto const max_alt =
      std::distance(begin(allocations),
                    std::max_element(begin(allocations), end(allocations)));
  return to_fbs(sched, fbb, cpg.alternatives_[max_alt].compact_journey_);
}

msg_ptr make_passenger_forecast_msg(
    schedule const& sched, motis::paxmon::paxmon_data const& data,
    std::vector<std::pair<combined_passenger_group*,
                          std::vector<std::uint16_t>>> const& cpg_allocations,
    simulation_result const& sim_result) {
  message_creator mc;
  std::vector<Offset<PassengerGroupForecast>> fb_groups;

  for (auto const& [cpg, allocations] : cpg_allocations) {

    auto const loc_type = fbs_localization_type(cpg->localization_);
    auto const loc = to_fbs(sched, mc, cpg->localization_);
    auto const forecast_journey =
        get_forecast_journey(sched, mc, *cpg, allocations);
    for (auto const& grp : cpg->groups_) {
      fb_groups.emplace_back(CreatePassengerGroupForecast(
          mc, to_fbs(sched, mc, *grp), loc_type, loc, forecast_journey));
    }
  }
  mc.create_and_finish(
      MsgContent_PassengerForecast,
      CreatePassengerForecast(mc, mc.CreateVector(fb_groups),
                              to_fbs(sched, mc, sim_result, data.graph_))
          .Union(),
      "/paxforecast/passenger_forecast");
  return make_msg(mc);
}

}  // namespace motis::paxforecast
