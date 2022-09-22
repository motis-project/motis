#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/group_route.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/service_info.h"
#include "motis/paxmon/temp_passenger_group.h"
#include "motis/paxmon/universe.h"

#include "motis/module/message.h"

namespace motis::paxmon {

flatbuffers::Offset<PaxMonCompactJourney> to_fbs(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    compact_journey const& cj);

flatbuffers::Offset<PaxMonCompactJourney> to_fbs(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    fws_compact_journey const& cj);

compact_journey from_fbs(schedule const& sched, PaxMonCompactJourney const* cj);

flatbuffers::Offset<PaxMonDataSource> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, data_source const& ds);

data_source from_fbs(PaxMonDataSource const* ds);

flatbuffers::Offset<PaxMonGroupRoute> to_fbs(
    schedule const& sched, passenger_group_container const& pgc,
    flatbuffers::FlatBufferBuilder& fbb, group_route const& gr);

flatbuffers::Offset<PaxMonGroupRoute> to_fbs(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    temp_group_route const& tgr);

temp_group_route from_fbs(schedule const& sched, PaxMonGroupRoute const* gr);

flatbuffers::Offset<PaxMonGroup> to_fbs(schedule const& sched,
                                        passenger_group_container const& pgc,
                                        flatbuffers::FlatBufferBuilder& fbb,
                                        passenger_group const& pg,
                                        bool with_reroute_log);

temp_passenger_group from_fbs(schedule const& sched, PaxMonGroup const* pg);

flatbuffers::Offset<PaxMonGroupWithRoute> to_fbs(
    schedule const& sched, passenger_group_container const& pgc,
    flatbuffers::FlatBufferBuilder& fbb,
    passenger_group_with_route const& pgwr);

flatbuffers::Offset<PaxMonGroupWithRoute> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb,
    temp_passenger_group_with_route const& tpgr);

temp_passenger_group_with_route from_fbs(schedule const& sched,
                                         PaxMonGroupWithRoute const* pgwr);

PaxMonGroupRouteBaseInfo to_fbs_base_info(
    flatbuffers::FlatBufferBuilder& /*fbb*/, passenger_group const& pg,
    group_route const& gr);

PaxMonGroupRouteBaseInfo to_fbs_base_info(
    flatbuffers::FlatBufferBuilder& /*fbb*/,
    passenger_group_container const& pgc,
    passenger_group_with_route const& pgwr);

PaxMonLocalization fbs_localization_type(passenger_localization const& loc);

flatbuffers::Offset<PaxMonLocalizationWrapper> to_fbs_localization_wrapper(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    passenger_localization const& loc);

flatbuffers::Offset<void> to_fbs(schedule const& sched,
                                 flatbuffers::FlatBufferBuilder& fbb,
                                 passenger_localization const& loc);

passenger_localization from_fbs(schedule const& sched,
                                PaxMonLocalization loc_type,
                                void const* loc_ptr);

passenger_localization from_fbs(schedule const& sched,
                                PaxMonLocalizationWrapper const* loc_wrapper);

flatbuffers::Offset<PaxMonEvent> to_fbs(schedule const& sched,
                                        passenger_group_container const& pgc,
                                        flatbuffers::FlatBufferBuilder& fbb,
                                        monitoring_event const& me);

flatbuffers::Offset<ServiceInfo> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                        service_info const& si);

flatbuffers::Offset<TripServiceInfo> to_fbs_trip_service_info(
    flatbuffers::FlatBufferBuilder& fbb, schedule const& sched, trip const* trp,
    std::vector<std::pair<service_info, unsigned>> const& service_infos);

flatbuffers::Offset<TripServiceInfo> to_fbs_trip_service_info(
    flatbuffers::FlatBufferBuilder& fbb, schedule const& sched,
    trip const* trp);

flatbuffers::Offset<TripServiceInfo> to_fbs_trip_service_info(
    flatbuffers::FlatBufferBuilder& fbb, schedule const& sched,
    journey_leg const& leg);

flatbuffers::Offset<PaxMonDistribution> to_fbs_distribution(
    flatbuffers::FlatBufferBuilder& fbb, pax_pdf const& pdf,
    pax_stats const& stats);

flatbuffers::Offset<PaxMonDistribution> to_fbs_distribution(
    flatbuffers::FlatBufferBuilder& fbb, pax_pdf const& pdf,
    pax_cdf const& cdf);

flatbuffers::Offset<PaxMonEdgeLoadInfo> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, schedule const& sched,
    universe const& uv, edge_load_info const& eli);

flatbuffers::Offset<PaxMonTripLoadInfo> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, schedule const& sched,
    universe const& uv, trip_load_info const& tli);

}  // namespace motis::paxmon
