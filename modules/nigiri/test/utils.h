#pragma once

#include <string>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

#include "motis/module/message.h"

namespace motis::nigiri {

struct trip_update {
  struct stop_time_update {
    std::string stop_id_;
    std::optional<std::size_t> seq_{std::nullopt};
    ::nigiri::event_type ev_type_{::nigiri::event_type::kDep};
    unsigned delay_minutes_{0U};
    bool skip_{false};
    std::optional<std::string> stop_assignment_{std::nullopt};
  };
  std::string trip_id_;
  std::vector<stop_time_update> stop_updates_{};
  bool cancelled_{false};
};

template <typename T>
std::int64_t to_unix(T&& x) {
  return std::chrono::time_point_cast<std::chrono::seconds>(x)
      .time_since_epoch()
      .count();
};

inline transit_realtime::FeedMessage to_feed_msg(
    std::vector<trip_update> const& trip_delays,
    date::sys_seconds const msg_time) {
  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(msg_time));

  auto id = 0U;
  for (auto const& trip : trip_delays) {
    auto const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    e->set_is_deleted(false);

    auto const td = e->mutable_trip_update()->mutable_trip();
    td->set_trip_id(trip.trip_id_);
    if (trip.cancelled_) {
      td->set_schedule_relationship(
          transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED);
      continue;
    }

    for (auto const& stop_upd : trip.stop_updates_) {
      auto* const upd = e->mutable_trip_update()->add_stop_time_update();
      if (!stop_upd.stop_id_.empty()) {
        *upd->mutable_stop_id() = stop_upd.stop_id_;
      }
      if (stop_upd.seq_.has_value()) {
        upd->set_stop_sequence(*stop_upd.seq_);
      }
      if (stop_upd.stop_assignment_.has_value()) {
        upd->mutable_stop_time_properties()->set_assigned_stop_id(
            stop_upd.stop_assignment_.value());
      }
      if (stop_upd.skip_) {
        upd->set_schedule_relationship(
            transit_realtime::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
        continue;
      }
      stop_upd.ev_type_ == ::nigiri::event_type::kDep
          ? upd->mutable_departure()->set_delay(stop_upd.delay_minutes_ * 60)
          : upd->mutable_arrival()->set_delay(stop_upd.delay_minutes_ * 60);
    }
  }

  return msg;
}

inline motis::module::msg_ptr make_routing_msg(std::string_view from,
                                               std::string_view to,
                                               std::int64_t const start) {
  using namespace motis;
  using flatbuffers::Offset;

  motis::module::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, motis::routing::Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              routing::CreateInputStation(fbb, fbb.CreateString(from),
                                          fbb.CreateString("")),
              start)
              .Union(),
          routing::CreateInputStation(fbb, fbb.CreateString(to),
                                      fbb.CreateString("")),
          routing::SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<routing::Via>>()),
          fbb.CreateVector(
              std::vector<Offset<routing::AdditionalEdgeWrapper>>()))
          .Union(),
      "/nigiri");
  return make_msg(fbb);
}

}  // namespace motis::nigiri