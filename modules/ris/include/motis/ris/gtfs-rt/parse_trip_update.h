#pragma once

#include <optional>

#include "flatbuffers/flatbuffers.h"

#include "motis/core/common/unixtime.h"
#include "motis/ris/gtfs-rt/common.h"

#include "motis/protocol/RISMessage_generated.h"
#include "gtfsrt.pb.h"

namespace motis {
struct schedule;
namespace ris::gtfsrt {

void handle_trip_update(
    trip_update_context& update_ctx, knowledge_context&, unixtime,
    std::function<void(message_context&, flatbuffers::Offset<Message>)> const&,
    std::string const& tag);

flatbuffers::Offset<Message> handle_unknown_cancl_trip_upd(
    schedule& sched, knowledge_context&, message_context&,
    transit_realtime::TripUpdate const&);

}  // namespace ris::gtfsrt
}  // namespace motis
