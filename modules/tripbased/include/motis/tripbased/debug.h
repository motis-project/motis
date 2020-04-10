#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/module.h"

#include "motis/tripbased/data.h"

namespace motis::tripbased {

flatbuffers::Offset<TripDebugInfo> get_trip_debug_info(
    flatbuffers::FlatBufferBuilder& fbb, tb_data const& data,
    schedule const& sched, TripSelectorWrapper const* selector);

flatbuffers::Offset<StationDebugInfo> get_station_debug_info(
    flatbuffers::FlatBufferBuilder& fbb, tb_data const& data,
    schedule const& sched, std::string const& eva_no);

}  // namespace motis::tripbased
