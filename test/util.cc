#include "./util.h"

#include "fmt/format.h"

namespace motis::test {

using namespace std::string_view_literals;
using namespace std::chrono_literals;
using namespace date;

using feed_entity = std::variant<trip_update, alert>;

transit_realtime::FeedMessage to_feed_msg(
    std::vector<feed_entity> const& feed_entities,
    date::sys_seconds const msg_time) {
  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(msg_time));

  auto id = 0U;
  for (auto const& x : feed_entities) {
    auto const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));

    auto const set_trip = [](::transit_realtime::TripDescriptor* td,
                             trip_descriptor const& trip,
                             bool const canceled = false) {
      td->set_trip_id(trip.trip_id_);
      if (canceled) {
        td->set_schedule_relationship(
            transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED);
        return;
      }
      if (trip.date_) {
        td->set_start_date(*trip.date_);
      }
      if (trip.start_time_) {
        td->set_start_time(*trip.start_time_);
      }
    };

    std::visit(
        utl::overloaded{
            [&](trip_update const& u) {
              set_trip(e->mutable_trip_update()->mutable_trip(), u.trip_,
                       u.cancelled_);

              for (auto const& stop_upd : u.stop_updates_) {
                auto* const upd =
                    e->mutable_trip_update()->add_stop_time_update();
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
                    ? upd->mutable_departure()->set_delay(
                          stop_upd.delay_minutes_ * 60)
                    : upd->mutable_arrival()->set_delay(
                          stop_upd.delay_minutes_ * 60);
              }
            },

            [&](alert const& a) {
              auto const alert = e->mutable_alert();

              auto const header =
                  alert->mutable_header_text()->add_translation();
              *header->mutable_text() = a.header_;
              *header->mutable_language() = "en";

              auto const description =
                  alert->mutable_description_text()->add_translation();
              *description->mutable_text() = a.description_;
              *description->mutable_language() = "en";

              for (auto const& entity : a.entities_) {
                auto const ie = alert->add_informed_entity();
                if (entity.agency_id_) {
                  *ie->mutable_agency_id() = *entity.agency_id_;
                }
                if (entity.route_id_) {
                  *ie->mutable_route_id() = *entity.route_id_;
                }
                if (entity.direction_id_) {
                  ie->set_direction_id(*entity.direction_id_);
                }
                if (entity.route_type_) {
                  ie->set_route_type(*entity.route_type_);
                }
                if (entity.stop_id_) {
                  ie->set_stop_id(*entity.stop_id_);
                }
                if (entity.trip_) {
                  set_trip(ie->mutable_trip(), *entity.trip_);
                }
              }
            }},
        x);
  }

  return msg;
}

}  // namespace motis::test