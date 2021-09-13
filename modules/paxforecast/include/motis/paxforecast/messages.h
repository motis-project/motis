#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/universe.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/load_forecast.h"
#include "motis/paxforecast/measures/measures.h"
#include "motis/paxforecast/simulation_result.h"

namespace motis::paxforecast {

motis::module::msg_ptr make_forecast_update_msg(
    schedule const& sched, motis::paxmon::universe const& uv,
    simulation_result const& sim_result, load_forecast const& lfc);

flatbuffers::Offset<Alternative> to_fbs(schedule const& sched,
                                        flatbuffers::FlatBufferBuilder& fbb,
                                        alternative const& alt);

measures::measures from_fbs(
    schedule const& sched,
    flatbuffers::Vector<flatbuffers::Offset<MeasureWrapper>> const* ms);

}  // namespace motis::paxforecast
