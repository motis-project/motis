#include "motis/ris/gtfs-rt/gtfsrt_parser.h"

#include <optional>

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/parse_time.h"
#include "motis/ris/gtfs-rt/parse_trip_update.h"

#ifdef CreateEvent
#undef CreateEvent
#endif

using namespace transit_realtime;
using namespace flatbuffers;
using namespace motis::logging;

namespace motis::ris::gtfsrt {

void finish_ris_msg(message_context& ctx, Offset<Message> message,
                    std::function<void(ris_message&&)> const& cb) {
  ctx.b_.Finish(message);
  auto ris_msg = ris_message(ctx.earliest_, ctx.latest_, ctx.timestamp_,
                             std::move(ctx.b_));
  cb(std::move(ris_msg));
}

void parse_trip_updates(schedule const& sched, knowledge_context& knowledge,
                        bool const is_additional_skip_allowed,
                        FeedEntity const& entity, unixtime const timestamp,
                        std::function<void(ris_message&&)> const& cb,
                        std::string const& tag) {
  auto trip_update = entity.trip_update();
  auto descriptor = trip_update.trip();
  switch (descriptor.schedule_relationship()) {
    case TripDescriptor_ScheduleRelationship_SCHEDULED:
    case TripDescriptor_ScheduleRelationship_ADDED:
    case TripDescriptor_ScheduleRelationship_CANCELED: {
      trip_update_context update_ctx{sched, trip_update,
                                     is_additional_skip_allowed};
      handle_trip_update(update_ctx, knowledge, timestamp,
                         [&](message_context& ctx, Offset<Message> msg) {
                           finish_ris_msg(ctx, msg, cb);
                         });
      break;
    }

    case TripDescriptor_ScheduleRelationship_UNSCHEDULED:
      throw std::runtime_error{
          "Found unsupported UNSCHEDULED trip update. Skipping."};

    default:
      throw std::runtime_error{
          "GTFS-RT found unhandled case for Schedule relationship"};
  }
}

void parse_entity(schedule const& sched, knowledge_context& knowledge,
                  bool const is_additional_skip_allowed,
                  FeedEntity const& entity, unixtime message_time,
                  std::function<void(ris_message&&)> const& cb,
                  std::string const& tag) {
  // every entity contains either a trip update, a vehicle update or an
  // alert.
  if (entity.has_trip_update()) {
    parse_trip_updates(sched, knowledge, is_additional_skip_allowed, entity,
                       message_time, cb, tag);
  } else if (entity.has_vehicle()) {
    LOG(logging::info) << "GTFS-RT Vehicle update not implemented.";
  } else if (entity.has_alert()) {
    LOG(logging::error) << "GTFS-RT Alert not implemented";
  } else {
    throw std::runtime_error{
        "GTFS-RT entity is neither a trip update, vehicle update or alert. "
        "Skipping."};
  }
}

void to_ris_message(schedule const& sched, knowledge_context& knowledge,
                    bool const is_additional_skip_allowed, std::string_view s,
                    std::function<void(ris_message&&)> const& cb,
                    std::string const& tag) {
  FeedMessage feed_message;

  auto const success = feed_message.ParseFromArray(
      reinterpret_cast<void const*>(s.data()), s.size());

  if (!success) {
    LOG(logging::error) << "GTFS-RT unable to parse protobuf message";
    return;
  }

  if (!feed_message.has_header()) {
    LOG(logging::error) << "GTFS-RT: skipping message without header";
    return;
  }

  LOG(info) << "parsing " << feed_message.entity().size() << " GTFS-RT updates";

  auto const message_time =
      static_cast<unixtime>(feed_message.header().timestamp());
  for (auto const& entity : feed_message.entity()) {
    try {
      parse_entity(sched, knowledge, is_additional_skip_allowed, entity,
                   message_time, cb, tag);
    } catch (const std::exception& e) {
      LOG(logging::error) << "Exception on entity " << entity.id()
                          << " for message with timestamp " << message_time
                          << ": " << e.what();
    }
  }
  knowledge.sort_known_lists();
}

std::vector<ris_message> parse(schedule const& sched,
                               knowledge_context& knowledge,
                               bool is_additional_skip_allowed,
                               std::string_view s, std::string const& tag) {
  std::vector<ris_message> msgs;
  to_ris_message(
      sched, knowledge, is_additional_skip_allowed, s,
      [&](ris_message&& m) { msgs.emplace_back(std::move(m)); }, tag);
  return msgs;
}

}  // namespace motis::ris::gtfsrt
