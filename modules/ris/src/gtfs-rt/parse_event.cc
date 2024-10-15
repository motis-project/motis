#include "motis/ris/gtfs-rt/parse_event.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/parse_stop.h"
#include "motis/ris/gtfs-rt/parse_time.h"

#ifdef CreateEvent
#undef CreateEvent
#endif

using namespace transit_realtime;
using namespace flatbuffers;

namespace motis::ris::gtfsrt {

Offset<IdEvent> create_id_event(message_context& ctx,
                                mcd::string const& station_id,
                                unixtime const start_time) {
  return CreateIdEvent(ctx.b_, ctx.b_.CreateString(station_id), 0, start_time,
                       IdEventType_Additional);
}

Offset<IdEvent> create_id_event(message_context& ctx, schedule const& sched,
                                trip const& trip) {
  auto const unix_time =
      motis_to_unixtime(sched.schedule_begin_, trip.id_.primary_.time_);
  auto const stop_id =
      sched.stations_.at(trip.id_.primary_.station_id_)->eva_nr_;
  auto const service_id = trip.id_.primary_.train_nr_;
  return CreateIdEvent(ctx.b_, ctx.b_.CreateString(stop_id), service_id,
                       unix_time, IdEventType_Schedule);
}

Offset<Event> create_event(trip const& trip, schedule& sched,
                           message_context& ctx, const int stop_idx,
                           const event_type type) {
  auto const stop = access::trip_stop{&trip, stop_idx};
  auto const stop_id = stop.get_station(sched).eva_nr_;
  auto const sched_time = get_schedule_time(trip, sched, stop_idx, type);
  return CreateEvent(
      ctx.b_, ctx.b_.CreateString(stop_id.view()), trip.id_.primary_.train_nr_,
      ctx.b_.CreateString(trip.id_.secondary_.line_id_.view()),
      type == event_type::DEP ? EventType_DEP : EventType_ARR, sched_time);
}

Offset<Event> create_event(trip const& trip, schedule& sched,
                           message_context& ctx, stop_context const& stop,
                           const event_type type) {
  return create_event(trip, sched, ctx, stop.idx_, type);
}

Offset<Event> create_event(message_context& ctx, evt const& event) {
  ctx.adjust_times(event.orig_sched_time_);
  return CreateEvent(
      ctx.b_, ctx.b_.CreateString(event.stop_id_), event.train_nr_,
      ctx.b_.CreateString(event.line_id_),
      event.type_ == event_type::ARR ? EventType_ARR : EventType_DEP,
      event.orig_sched_time_);
}

Offset<Message> create_delay_message(message_context& ctx,
                                     Offset<IdEvent> const& id_event,
                                     std::vector<evt> const& delay_evts,
                                     DelayType delay) {
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_DelayMessage,
      CreateDelayMessage(ctx.b_, id_event, delay,
                         ctx.b_.CreateVector(utl::to_vec(
                             delay_evts,
                             [&](evt const& event) {
                               ctx.adjust_times(event.new_sched_time_);
                               return CreateUpdatedEvent(
                                   ctx.b_, create_event(ctx, event),
                                   event.new_sched_time_);
                             })))
          .Union());
}

Offset<Message> create_reroute_msg(message_context& ctx,
                                   Offset<IdEvent> const& id_event,
                                   std::vector<evt> const& reroute_evts) {
  std::vector<Offset<Event>> events;
  std::for_each(begin(reroute_evts), end(reroute_evts), [&](evt const& event) {
    events.emplace_back(create_event(ctx, event));
  });
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_RerouteMessage,
      CreateRerouteMessage(
          ctx.b_, id_event, ctx.b_.CreateVector(events),
          ctx.b_.CreateVector(std::vector<Offset<ReroutedEvent>>{}))
          .Union());
}

Offset<Message> create_cancel_msg(message_context& ctx,
                                  Offset<IdEvent> const& id_event,
                                  std::vector<evt> const& reroute_evts) {
  std::vector<Offset<Event>> events;
  std::for_each(begin(reroute_evts), end(reroute_evts), [&](evt const& event) {
    events.emplace_back(create_event(ctx, event));
  });
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_CancelMessage,
      CreateCancelMessage(ctx.b_, id_event, ctx.b_.CreateVector(events))
          .Union());
}

Offset<Message> create_additional_msg(message_context& ctx,
                                      Offset<IdEvent> const& id_event,
                                      std::vector<evt> const& additional_evts) {
  std::vector<Offset<AdditionalEvent>> events;
  std::for_each(
      begin(additional_evts), end(additional_evts), [&](evt const& event) {
        events.emplace_back(
            CreateAdditionalEvent(ctx.b_, create_event(ctx, event),
                                  ctx.b_.CreateString("Additional Service"),
                                  ctx.b_.CreateString(""), event.seq_no_));
      });
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_AdditionMessage,
      CreateAdditionMessage(ctx.b_, id_event, ctx.b_.CreateVector(events))
          .Union());
}

}  // namespace motis::ris::gtfsrt
