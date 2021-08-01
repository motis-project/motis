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
    trip_update_context& update_ctx, std::unique_ptr<knowledge_context> const&,
    unixtime,
    std::function<void(message_context&, flatbuffers::Offset<Message>)> const&);

flatbuffers::Offset<Message> handle_unknown_cancl_trip_upd(
    schedule& sched, std::unique_ptr<knowledge_context> const&,
    message_context&, transit_realtime::TripUpdate const&);

}  // namespace ris::gtfsrt
}  // namespace motis
