#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/protocol/Message_generated.h"

namespace motis::lookup {

flatbuffers::Offset<LookupRiBasisResponse> lookup_ribasis(
    flatbuffers::FlatBufferBuilder&, schedule const&,
    LookupRiBasisRequest const*);

}  // namespace motis::lookup
