#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/rsl/combined_passenger_group.h"
#include "motis/rsl/monitoring_event.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/simulation_result.h"

#include "motis/module/message.h"

namespace motis::rsl {

flatbuffers::Offset<CompactJourney> to_fbs(schedule const& sched,
                                           flatbuffers::FlatBufferBuilder& fbb,
                                           compact_journey const& cj);

flatbuffers::Offset<PassengerGroup> to_fbs(schedule const& sched,
                                           flatbuffers::FlatBufferBuilder& fbb,
                                           passenger_group const& pg);

flatbuffers::Offset<void> to_fbs(schedule const& sched,
                                 flatbuffers::FlatBufferBuilder& fbb,
                                 passenger_localization const& loc);

flatbuffers::Offset<MonitoringEvent> to_fbs(schedule const& sched,
                                            flatbuffers::FlatBufferBuilder& fbb,
                                            monitoring_event const& me);

motis::module::msg_ptr make_passenger_forecast_msg(
    schedule const& sched, rsl_data const& data,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    simulation_result const& sim_result);

}  // namespace motis::rsl
