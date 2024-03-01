#pragma once

#include <vector>

#include "motis/module/message.h"

#include "motis/core/journey/journey.h"

namespace motis {

struct schedule;

flatbuffers::Offset<Connection> to_connection(
    flatbuffers::FlatBufferBuilder&, journey const&,
    bool include_invalid_stop_info = false);

}  // namespace motis
