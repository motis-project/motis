#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/paxmon_data.h"

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

}  // namespace motis::paxmon
