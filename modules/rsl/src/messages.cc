#include "motis/rsl/messages.h"

#include "utl/to_vec.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

using namespace motis::module;
using namespace flatbuffers;

namespace motis::rsl {

Offset<RslTransferInfo> to_fbs(FlatBufferBuilder& fbb,
                               std::optional<transfer_info> const& ti) {
  if (ti) {
    auto const& val = ti.value();
    return CreateRslTransferInfo(fbb,
                                 val.type_ == transfer_info::type::SAME_STATION
                                     ? RslTransferType_SAME_STATION
                                     : RslTransferType_FOOTPATH,
                                 val.duration_);
  } else {
    return CreateRslTransferInfo(fbb, RslTransferType_NONE);
  }
}

Offset<RslJourneyLeg> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                             journey_leg const& leg) {
  return CreateRslJourneyLeg(
      fbb, to_fbs(fbb, leg.trip_),
      to_fbs(fbb, *sched.stations_[leg.enter_station_id_]),
      to_fbs(fbb, *sched.stations_[leg.exit_station_id_]),
      motis_to_unixtime(sched, leg.enter_time_),
      motis_to_unixtime(sched, leg.exit_time_),
      to_fbs(fbb, leg.enter_transfer_));
}

Offset<RslJourney> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                          compact_journey const& cj) {
  return CreateRslJourney(
      fbb, fbb.CreateVector(utl::to_vec(cj.legs_, [&](journey_leg const& leg) {
        return to_fbs(sched, fbb, leg);
      })));
}

Offset<RslPassengerGroup> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                                 passenger_group const& pg) {
  return CreateRslPassengerGroup(
      fbb, 0, pg.passengers_, to_fbs(sched, fbb, pg.compact_planned_journey_));
}

Offset<void> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                    passenger_localization const& loc) {
  if (loc.in_trip()) {
    return CreateRslInTrip(fbb, to_fbs(sched, fbb, loc.in_trip_),
                           to_fbs(fbb, *loc.at_station_),
                           motis_to_unixtime(sched, loc.arrival_time_))
        .Union();
  } else {
    return CreateRslAtStation(fbb, to_fbs(fbb, *loc.at_station_),
                              motis_to_unixtime(sched, loc.arrival_time_))
        .Union();
  }
}

Offset<RslAlternative> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                              alternative const& alt) {
  return CreateRslAlternative(fbb, to_fbs(sched, fbb, alt.compact_journey_),
                              motis_to_unixtime(sched, alt.arrival_time_),
                              alt.duration_, alt.transfers_);
}

Offset<RslCombinedPassengerGroup> to_fbs(schedule const& sched,
                                         FlatBufferBuilder& fbb,
                                         combined_passenger_group const& cpg) {
  return CreateRslCombinedPassengerGroup(
      fbb, cpg.passengers_,
      cpg.localization_.in_trip() ? RslPassengerLocalization_RslInTrip
                                  : RslPassengerLocalization_RslAtStation,
      to_fbs(sched, fbb, cpg.localization_),
      to_fbs(fbb, *sched.stations_[cpg.destination_station_id_]),
      fbb.CreateVector(utl::to_vec(
          cpg.groups_,
          [&](passenger_group* const pg) { return to_fbs(sched, fbb, *pg); })),
      fbb.CreateVector(utl::to_vec(
          cpg.alternatives_,
          [&](alternative const& alt) { return to_fbs(sched, fbb, alt); })));
}

Offset<RslSimResult> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                            simulation_result const& res, graph const& g) {
  auto const trip_with_edges_to_fbs = [&](trip const* trp,
                                          std::vector<edge*> const& edges) {
    std::vector<Offset<RslEdgeOverCapacity>> fb_edges;
    for (auto const e : edges) {
      if (e->type_ != edge_type::TRIP) {
        continue;
      }
      fb_edges.emplace_back(CreateRslEdgeOverCapacity(
          fbb, e->passengers_, e->capacity_, res.additional_passengers_.at(e),
          to_fbs(fbb, *sched.stations_[e->from(g)->station_]),
          to_fbs(fbb, *sched.stations_[e->to(g)->station_])));
    }
    return CreateRslTripOverCapacity(fbb, to_fbs(sched, fbb, trp),
                                     fbb.CreateVector(fb_edges));
  };

  return CreateRslSimResult(
      fbb, res.is_over_capacity(), res.edge_count_over_capacity(),
      res.total_passengers_over_capacity(),
      fbb.CreateVector(utl::to_vec(
          res.trips_over_capacity_with_edges(), [&](auto const& kv) {
            return trip_with_edges_to_fbs(kv.first, kv.second);
          })));
}

msg_ptr make_journeys_broken_msg(
    schedule const& sched, rsl_data const& data,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    simulation_result const& sim_result) {
  message_creator mc;
  std::vector<Offset<RslCombinedPassengerGroup>> fb_cpgs;
  for (auto const& [destination_station_id, cpgs] : combined_groups) {
    (void)destination_station_id;
    for (auto const& cpg : cpgs) {
      fb_cpgs.emplace_back(to_fbs(sched, mc, cpg));
    }
  }
  mc.create_and_finish(
      MsgContent_RslJourneysBroken,
      CreateRslJourneysBroken(mc, sched.system_time_, mc.CreateVector(fb_cpgs),
                              to_fbs(sched, mc, sim_result, data.graph_))
          .Union(),
      "/rsl/journeys_broken");
  return make_msg(mc);
}

}  // namespace motis::rsl
