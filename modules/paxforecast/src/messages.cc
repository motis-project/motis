#include "motis/paxforecast/messages.h"

#include "utl/to_vec.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace flatbuffers;
using namespace motis::paxmon;

namespace motis::paxforecast {

Offset<PassengerGroupForecast> get_passenger_group_forecast(
    FlatBufferBuilder& fbb, schedule const& sched, passenger_group const& grp,
    group_simulation_result const& group_result) {
  return CreatePassengerGroupForecast(
      fbb, to_fbs(sched, fbb, grp),
      fbs_localization_type(*group_result.localization_),
      to_fbs(sched, fbb, *group_result.localization_),
      fbb.CreateVector(
          utl::to_vec(group_result.alternatives_, [&](auto const& alt) {
            return CreateForecastAlternative(
                fbb, to_fbs(sched, fbb, alt.first->compact_journey_),
                alt.second);
          })));
}

Offset<EdgeOverCapacity> get_edge_over_capacity(
    FlatBufferBuilder& fbb, schedule const& sched, graph const& g,
    motis::paxmon::edge const& e, edge_over_capacity_info const& oci) {
  auto const from = e.from(g);
  auto const to = e.to(g);
  return CreateEdgeOverCapacity(fbb, oci.current_pax_ + oci.additional_pax_,
                                e.capacity(), oci.additional_pax_,
                                to_fbs(fbb, from->get_station(sched)),
                                to_fbs(fbb, to->get_station(sched)),
                                motis_to_unixtime(sched, from->schedule_time()),
                                motis_to_unixtime(sched, to->schedule_time()));
}

Offset<OverCapacityInfo> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                                graph const& g, over_capacity_info const& oci) {
  return CreateOverCapacityInfo(
      fbb, oci.probability_, oci.over_capacity_edges_.size(),
      fbb.CreateVector(
          utl::to_vec(oci.over_capacity_trips_, [&](auto const& entry) {
            return CreateTripOverCapacity(
                fbb, to_fbs(sched, fbb, entry.first),
                fbb.CreateVector(utl::to_vec(entry.second, [&](auto const* e) {
                  return get_edge_over_capacity(fbb, sched, g, *e,
                                                oci.over_capacity_edges_.at(e));
                })));
          })));
}

msg_ptr make_passenger_forecast_msg(
    schedule const& sched, motis::paxmon::paxmon_data const& data,
    simulation_result const& sim_result,
    std::vector<over_capacity_info> const& over_capacity_infos) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_PassengerForecast,
      CreatePassengerForecast(
          fbb,
          fbb.CreateVector(utl::to_vec(sim_result.group_results_,
                                       [&](auto const& entry) {
                                         return get_passenger_group_forecast(
                                             fbb, sched, *entry.first,
                                             entry.second);
                                       })),
          fbb.CreateVector(utl::to_vec(over_capacity_infos,
                                       [&](auto const& oci) {
                                         return to_fbs(fbb, sched, data.graph_,
                                                       oci);
                                       })))
          .Union(),
      "/paxforecast/passenger_forecast");
  return make_msg(fbb);
}

}  // namespace motis::paxforecast
