#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/protocol/Message_generated.h"

namespace motis::lookup {

flatbuffers::Offset<LookupMetaStationResponse> lookup_meta_station(
    flatbuffers::FlatBufferBuilder&, schedule const&,
    LookupMetaStationRequest const*);

}  // namespace motis::lookup
