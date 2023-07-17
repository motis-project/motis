#pragma once

#include <string>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

#include "motis/module/message.h"

namespace motis::nigiri {

struct trip {
  struct delay {
    std::string stop_id_;
    ::nigiri::event_type ev_type_;
    unsigned delay_minutes_{0U};
  };
  std::string trip_id_;
  std::vector<delay> delays_;
};

template <typename T>
std::int64_t to_unix(T&& x) {
  return std::chrono::time_point_cast<std::chrono::seconds>(x)
      .time_since_epoch()
      .count();
};

inline transit_realtime::FeedMessage to_feed_msg(
    std::vector<trip> const& trip_delays, date::sys_seconds const msg_time) {
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

    for (auto const& stop_delay : trip.delays_) {
      auto* const upd = e->mutable_trip_update()->add_stop_time_update();
      *upd->mutable_stop_id() = stop_delay.stop_id_;
      stop_delay.ev_type_ == ::nigiri::event_type::kDep
          ? upd->mutable_departure()->set_delay(stop_delay.delay_minutes_ * 60)
          : upd->mutable_arrival()->set_delay(stop_delay.delay_minutes_ * 60);
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