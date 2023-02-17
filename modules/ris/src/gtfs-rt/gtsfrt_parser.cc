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

void parse_trip_updates(knowledge_context& knowledge,
                        bool const is_additional_skip_allowed,
                        FeedEntity const& entity, unixtime const timestamp,
                        std::function<void(ris_message&&)> const& cb,
                        std::string const& tag) {
  auto trip_update = entity.trip_update();
  auto descriptor = trip_update.trip();
  switch (descriptor.schedule_relationship()) {
    case TripDescriptor_ScheduleRelationship_SCHEDULED:
    case TripDescriptor_ScheduleRelationship_ADDED:
    case TripDescriptor_ScheduleRelationship_UNSCHEDULED:
    case TripDescriptor_ScheduleRelationship_CANCELED: {
      trip_update_context update_ctx{knowledge.sched_, trip_update,
                                     is_additional_skip_allowed};
      handle_trip_update(
          update_ctx, knowledge, timestamp,
          [&](message_context& ctx, Offset<Message> msg) {
            finish_ris_msg(ctx, msg, cb);
          },
          tag);
      break;
    }

    case TripDescriptor_ScheduleRelationship_DUPLICATED:
    default:
      throw utl::fail("unhandled schedule relationship {}",
                      TripDescriptor_ScheduleRelationship_Name(
                          descriptor.schedule_relationship()));
  }
}

void parse_entity(knowledge_context& knowledge,
                  bool const is_additional_skip_allowed,
                  FeedEntity const& entity, unixtime message_time,
                  std::function<void(ris_message&&)> const& cb,
                  std::string const& tag) {
  if (entity.has_trip_update()) {
    parse_trip_updates(knowledge, is_additional_skip_allowed, entity,
                       message_time, cb, tag);
  }
}

void to_ris_message(knowledge_context& knowledge,
                    bool const is_additional_skip_allowed, std::string_view s,
                    std::function<void(ris_message&&)> const& cb,
                    std::string const& tag) {
  FeedMessage feed_message;

  auto const success = feed_message.ParseFromArray(
      reinterpret_cast<void const*>(s.data()), s.size());

  if (!success) {
    LOG(logging::error) << "GTFS-RT unable to parse protobuf message " << tag
                        << ": \""
                        << s.substr(0, std::min(s.size(), size_t{1000U}))
                        << (s.size() > 1000U ? "..." : "") << "\"";
    return;
  }

  if (!feed_message.has_header()) {
    LOG(logging::error) << "GTFS-RT: skipping message without header" << tag;
    return;
  }

  auto successful = 0U;
  auto const message_time =
      static_cast<unixtime>(feed_message.header().timestamp());
  for (auto const& entity : feed_message.entity()) {
    try {
      parse_entity(knowledge, is_additional_skip_allowed, entity, message_time,
                   cb, tag);
      ++successful;
    } catch (const std::exception& e) {
      LOG(logging::error) << "Exception on entity " << entity.id()
                          << " for message with timestamp " << message_time
                          << ": " << e.what()
                          << ", message=" << entity.DebugString();
    }
  }
  LOG(info) << (tag.empty() ? "" : tag.substr(0, tag.size() - 1) + ": ")
            << "parsed " << feed_message.entity().size()
            << " GTFS-RT updates: " << successful << "/"
            << feed_message.entity().size() << " ("
            << (100.0 * successful / feed_message.entity().size()) << "%)"
            << " successful";
  knowledge.sort_known_lists();
}

std::vector<ris_message> parse(knowledge_context& knowledge,
                               bool is_additional_skip_allowed,
                               std::string_view s, std::string const& tag) {
  std::vector<ris_message> msgs;
  to_ris_message(
      knowledge, is_additional_skip_allowed, s,
      [&](ris_message&& m) { msgs.emplace_back(std::move(m)); }, tag);
  return msgs;
}

}  // namespace motis::ris::gtfsrt
