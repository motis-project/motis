#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/load_info.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/service_info.h"

#include "motis/module/message.h"

namespace motis::paxmon {

flatbuffers::Offset<CompactJourney> to_fbs(schedule const& sched,
                                           flatbuffers::FlatBufferBuilder& fbb,
                                           compact_journey const& cj);

compact_journey from_fbs(schedule const& sched, CompactJourney const* cj);

flatbuffers::Offset<PassengerGroup> to_fbs(schedule const& sched,
                                           flatbuffers::FlatBufferBuilder& fbb,
                                           passenger_group const& pg);

passenger_group from_fbs(schedule const& sched, PassengerGroup const* pg);

PassengerLocalization fbs_localization_type(passenger_localization const& loc);

flatbuffers::Offset<void> to_fbs(schedule const& sched,
                                 flatbuffers::FlatBufferBuilder& fbb,
                                 passenger_localization const& loc);

passenger_localization from_fbs(schedule const& sched,
                                PassengerLocalization loc_type,
                                void const* loc_ptr);

flatbuffers::Offset<MonitoringEvent> to_fbs(schedule const& sched,
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

flatbuffers::Offset<EdgeLoadInfo> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                         schedule const& sched, graph const& g,
                                         edge_load_info const& eli);

flatbuffers::Offset<TripLoadInfo> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                         schedule const& sched, graph const& g,
                                         trip_load_info const& tli);

}  // namespace motis::paxmon
