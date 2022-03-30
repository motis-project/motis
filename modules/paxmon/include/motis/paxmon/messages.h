#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/load_info.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/service_info.h"
#include "motis/paxmon/universe.h"

#include "motis/module/message.h"

namespace motis::paxmon {

flatbuffers::Offset<PaxMonCompactJourney> to_fbs(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    compact_journey const& cj);

compact_journey from_fbs(schedule const& sched, PaxMonCompactJourney const* cj);

flatbuffers::Offset<PaxMonDataSource> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, data_source const& ds);

data_source from_fbs(PaxMonDataSource const* ds);

flatbuffers::Offset<PaxMonGroup> to_fbs(schedule const& sched,
                                        flatbuffers::FlatBufferBuilder& fbb,
                                        passenger_group const& pg);

passenger_group from_fbs(schedule const& sched, PaxMonGroup const* pg);

PaxMonGroupBaseInfo to_fbs_base_info(flatbuffers::FlatBufferBuilder& fbb,
                                     passenger_group const& pg);

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
