#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/universe.h"

#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast {

measures::measure_collection from_fbs(
    schedule const& sched,
    flatbuffers::Vector<flatbuffers::Offset<MeasureWrapper>> const* ms);

}  // namespace motis::paxforecast
