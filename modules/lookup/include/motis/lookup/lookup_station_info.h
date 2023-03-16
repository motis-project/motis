#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/protocol/Message_generated.h"

namespace motis::lookup {

flatbuffers::Offset<LookupStationInfoResponse> lookup_station_info(
    flatbuffers::FlatBufferBuilder&, schedule const&,
    LookupStationInfoRequest const*);

}  // namespace motis::lookup
