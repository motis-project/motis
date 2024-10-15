#pragma once

#include <ctime>

#include "motis/string.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event_type.h"

#include "motis/protocol/RISMessage_generated.h"
#include "gtfsrt.pb.h"

namespace motis {
struct trip;
struct schedule;
namespace ris::gtfsrt {

struct message_context;
struct stop_context;
struct evt;

flatbuffers::Offset<IdEvent> create_id_event(message_context&, schedule const&,
                                             trip const&);

flatbuffers::Offset<IdEvent> create_id_event(message_context&,
                                             mcd::string const& station_id,
                                             unixtime start_time);

flatbuffers::Offset<Event> create_event(trip const&, schedule const&,
                                        message_context&, stop_context const&,
                                        event_type);
flatbuffers::Offset<Event> create_event(trip const&, schedule const&,
                                        message_context&, int, event_type);

flatbuffers::Offset<Event> create_event(message_context&, evt const&);

flatbuffers::Offset<Message> create_delay_message(
    message_context&, flatbuffers::Offset<IdEvent> const&,
    std::vector<evt> const&, motis::ris::DelayType);

flatbuffers::Offset<Message> create_reroute_msg(
    message_context&, flatbuffers::Offset<IdEvent> const&,
    std::vector<evt> const&);

flatbuffers::Offset<Message> create_cancel_msg(
    message_context&, flatbuffers::Offset<IdEvent> const&,
    std::vector<evt> const&);

flatbuffers::Offset<Message> create_additional_msg(
    message_context&, flatbuffers::Offset<IdEvent> const&,
    std::vector<evt> const&);

}  // namespace ris::gtfsrt
}  // namespace motis
