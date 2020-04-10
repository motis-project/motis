#include "motis/ris/gtfs-rt/gtfsrt_parser.h"

#include <optional>

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/module/context/get_schedule.h"
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
using namespace motis::module;

namespace motis::ris::gtfsrt {

void finish_ris_msg(message_context& ctx, Offset<Message> message,
                    std::function<void(ris_message&&)> const& cb) {
  ctx.b_.Finish(message);
  auto ris_msg = ris_message(ctx.earliest_, ctx.latest_, ctx.timestamp_,
                             std::move(ctx.b_));
  cb(std::move(ris_msg));
}

void gtfsrt_parser::parse_trip_updates(
    schedule& sched, FeedEntity const& entity, std::time_t const timestamp,
    std::function<void(ris_message&&)> const& cb) {
  auto trip_update = entity.trip_update();
  auto descriptor = trip_update.trip();
  switch (descriptor.schedule_relationship()) {
    case TripDescriptor_ScheduleRelationship_SCHEDULED:
    case TripDescriptor_ScheduleRelationship_ADDED:
    case TripDescriptor_ScheduleRelationship_CANCELED: {
      trip_update_context update_ctx{sched, trip_update,
                                     is_addition_skip_allowed_};
      handle_trip_update(update_ctx, knowledge_, timestamp,
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

void gtfsrt_parser::parse_entity(schedule& sched, FeedEntity const& entity,
                                 std::time_t message_time,
                                 std::function<void(ris_message&&)> const& cb) {
  // every entity contains either a trip update, a vehicle update or an
  // alert.
  if (entity.has_trip_update()) {
    parse_trip_updates(sched, entity, message_time, cb);
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

void gtfsrt_parser::to_ris_message(
    std::string_view s, std::function<void(ris_message&&)> const& cb) {
  to_ris_message(get_schedule(), s, cb);
}

void gtfsrt_parser::to_ris_message(
    schedule& sched, std::string_view s,
    std::function<void(ris_message&&)> const& cb) {
  FeedMessage feed_message;

  bool success = feed_message.ParseFromArray(
      reinterpret_cast<void const*>(s.data()), s.size());

  if (!success) {
    LOG(logging::error)
        << "Failed to parse gtfs-rt message from protocol buffer!";
    return;
  }

  // check for header
  if (!feed_message.has_header()) {
    LOG(logging::error)
        << "GTFS-RT: Feed Message does not contain header. Skipping.";
    return;
  }

  // Check GTFS-RT version
  auto const& header = feed_message.header();
  if (header.gtfs_realtime_version() != "1.0" &&
      header.gtfs_realtime_version() != "2.0") {
    throw std::runtime_error{
        "Found unsupported GTFS-RT version. Supported is only 1.0 or 2.0"};
  }

  // Check dataset is build fully instead of incremental
  if (header.incrementality() != FeedHeader_Incrementality_FULL_DATASET) {
    throw std::runtime_error{
        "Found incrementallity level other than FULL_DATASET. This is not "
        "supported."};
  }

  LOG(info) << "Parsing " << feed_message.entity().size() << " GTFS-RT updates";

  std::time_t message_time{
      static_cast<std::time_t>(feed_message.header().timestamp())};
  for (auto const& entity : feed_message.entity()) {
    try {
      parse_entity(sched, entity, message_time, cb);
    } catch (const std::exception& e) {
      LOG(logging::error) << "Exception on entity " << entity.id()
                          << " for message with timestamp " << message_time
                          << ": " << e.what();
    }
  }
  knowledge_->sort_known_lists();

  static int count = 0;
  LOG(logging::error) << "Finished parsing Message: " << ++count;
}

std::vector<ris_message> gtfsrt_parser::parse(std::string_view s) {
  return parse(get_schedule(), s);
}

std::vector<ris_message> gtfsrt_parser::parse(schedule& sched,
                                              std::string_view s) {
  std::vector<ris_message> msgs;
  to_ris_message(sched, s,
                 [&](ris_message&& m) { msgs.emplace_back(std::move(m)); });
  return msgs;
}

gtfsrt_parser::gtfsrt_parser()
    : knowledge_(std::make_unique<knowledge_context>()) {}

gtfsrt_parser::~gtfsrt_parser() = default;

}  // namespace motis::ris::gtfsrt