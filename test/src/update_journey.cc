#include "motis/test/schedule/update_journey.h"

#include "flatbuffers/flatbuffers.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/event_type_conv.h"

#if defined(CreateEvent)
#undef CreateEvent
#endif

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::ris;
using namespace motis::routing;

namespace motis::test::schedule::update_journey {

void create_free_text_msg(FlatBufferBuilder& fbb, std::string const& eva_num,
                          int const service_num, event_type const type,
                          time_t const schedule_time,
                          std::string const& trip_start_eva,
                          time_t const trip_start_schedule_time,
                          int const free_text_code,
                          std::string const& free_text_text,
                          std::string const& free_text_type) {
  // clang-format off
  std::vector<Offset<Event>> const events{
        CreateEvent(fbb,
          fbb.CreateString(eva_num),
          service_num,
          fbb.CreateString(""),
          to_fbs(type),
          schedule_time
        )
  };
  auto const trip_id = CreateIdEvent(fbb,
        fbb.CreateString(trip_start_eva),
        service_num,
        trip_start_schedule_time);
  auto r = Range(0,0);
  auto const free_text = CreateFreeText(fbb,
        &r,
        free_text_code,
        fbb.CreateString(free_text_text), fbb.CreateString(free_text_type));
  // clang-format on
  fbb.Finish(CreateMessage(
      fbb, 0, 0, 0, MessageUnion_FreeTextMessage,
      CreateFreeTextMessage(fbb, trip_id, fbb.CreateVector(events), free_text)
          .Union()));
}

msg_ptr get_free_text_ris_msg(std::string const& eva_num, int const service_num,
                              event_type const type, time_t const schedule_time,
                              std::string const& trip_start_eva,
                              time_t const trip_start_schedule_time,
                              int const free_text_code,
                              std::string const& free_text_text,
                              std::string const& free_text_type) {
  FlatBufferBuilder is_msg_fbb;
  create_free_text_msg(is_msg_fbb, eva_num, service_num, type, schedule_time,
                       trip_start_eva, trip_start_schedule_time, free_text_code,
                       free_text_text, free_text_type);
  message_creator mc;
  std::vector<Offset<MessageHolder>> const messages{CreateMessageHolder(
      mc,
      mc.CreateVector(is_msg_fbb.GetBufferPointer(), is_msg_fbb.GetSize()))};
  mc.create_and_finish(MsgContent_RISBatch,
                       CreateRISBatch(mc, mc.CreateVector(messages)).Union(),
                       "/ris/messages", DestinationType_Topic);
  return make_msg(mc);
}

msg_ptr get_routing_request(time_t const from, time_t const to,
                            std::string const& eva_from,
                            std::string const& eva_to) {
  message_creator fbb;
  auto const interval = Interval{from, to};
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(fbb,
                             CreateInputStation(fbb, fbb.CreateString(eva_from),
                                                fbb.CreateString("")),
                             &interval, 0, false, false)
              .Union(),
          CreateInputStation(fbb, fbb.CreateString(eva_to),
                             fbb.CreateString("")),
          SearchType_SingleCriterion, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>{}),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{}))
          .Union(),
      "/routing", DestinationType_Topic);
  return make_msg(fbb);
}

void create_station_event_time_change_msg(
    FlatBufferBuilder& fbb, std::string const& eva_num, int const service_num,
    event_type const type, time_t const schedule_time, time_t const update_time,
    std::string const& trip_start_eva, time_t const trip_start_schedule_time) {
  // clang-format off
  std::vector<Offset<UpdatedEvent>> const events{
    CreateUpdatedEvent(fbb,
        CreateEvent(fbb,
          fbb.CreateString(eva_num), service_num,
          fbb.CreateString(""),
                  to_fbs(type),
                  schedule_time
        ),
                update_time
    )
  };
  auto const trip_id = CreateIdEvent(fbb,
        fbb.CreateString(trip_start_eva),
                service_num,
                trip_start_schedule_time);
  // clang-format on

  fbb.Finish(CreateMessage(
      fbb, 0, 0, 0, MessageUnion_DelayMessage,
      CreateDelayMessage(fbb, trip_id, DelayType_Is, fbb.CreateVector(events))
          .Union()));
}

msg_ptr get_delay_ris_msg(std::string const& eva_num, int const service_num,
                          event_type const type, time_t const schedule_time,
                          time_t const update_time,
                          std::string const& trip_start_eva,
                          time_t const trip_start_schedule_time) {
  FlatBufferBuilder is_msg_fbb;
  create_station_event_time_change_msg(
      is_msg_fbb, eva_num, service_num, type, schedule_time, update_time,
      trip_start_eva, trip_start_schedule_time);
  message_creator mc;
  std::vector<Offset<MessageHolder>> const messages{CreateMessageHolder(
      mc,
      mc.CreateVector(is_msg_fbb.GetBufferPointer(), is_msg_fbb.GetSize()))};
  mc.create_and_finish(MsgContent_RISBatch,
                       CreateRISBatch(mc, mc.CreateVector(messages)).Union(),
                       "/ris/messages", DestinationType_Topic);
  return make_msg(mc);
}

void create_canceled_train_msg(
    FlatBufferBuilder& fbb, std::string const& eva_num, int const service_num,
    event_type const type, time_t const schedule_time,
    std::string const& eva_num1, event_type const type1,
    time_t const schedule_time1, std::string const& trip_start_eva,
    time_t const trip_start_schedule_time) {
  // clang-format off
  std::vector<Offset<Event>> const events{
        CreateEvent(fbb,
          fbb.CreateString(eva_num),
          service_num,
          fbb.CreateString(""),
          to_fbs(type),
          schedule_time
        ),
        CreateEvent(fbb,
          fbb.CreateString(eva_num1),
          service_num,
          fbb.CreateString(""),
          to_fbs(type1),
          schedule_time1
        )
  };
  auto const trip_id = CreateIdEvent(fbb,
        fbb.CreateString(trip_start_eva),
        service_num,
        trip_start_schedule_time);
  // clang-format on
  fbb.Finish(CreateMessage(
      fbb, 0, 0, 0, MessageUnion_CancelMessage,
      CreateCancelMessage(fbb, trip_id, fbb.CreateVector(events)).Union()));
}

msg_ptr get_canceled_train_ris_message(
    std::string const& eva_num, int const service_num, event_type const type,
    time_t const schedule_time, std::string const& eva_num1,
    event_type const type1, time_t const schedule_time1,
    std::string const& trip_start_eva, time_t const trip_start_schedule_time) {
  FlatBufferBuilder is_msg_fbb;
  create_canceled_train_msg(is_msg_fbb, eva_num, service_num, type,
                            schedule_time, eva_num1, type1, schedule_time1,
                            trip_start_eva, trip_start_schedule_time);
  message_creator mc;
  std::vector<Offset<MessageHolder>> const messages{CreateMessageHolder(
      mc,
      mc.CreateVector(is_msg_fbb.GetBufferPointer(), is_msg_fbb.GetSize()))};
  mc.create_and_finish(MsgContent_RISBatch,
                       CreateRISBatch(mc, mc.CreateVector(messages)).Union(),
                       "/ris/messages", DestinationType_Topic);
  return make_msg(mc);
}

void create_track_msg(FlatBufferBuilder& fbb, std::string const& eva_num,
                      int const service_num, event_type const type,
                      time_t const schedule_time,
                      std::string const& trip_start_eva,
                      time_t const trip_start_schedule_time,
                      std::string const& updated_track) {
  // clang-format off
      std::vector<Offset<UpdatedTrack>> const events{
            CreateUpdatedTrack(fbb,CreateEvent(fbb,
              fbb.CreateString(eva_num),
              service_num,
              fbb.CreateString(""),
              to_fbs(type),
              schedule_time
            ),
            fbb.CreateString(updated_track))
      };
      auto const trip_id = CreateIdEvent(fbb,
            fbb.CreateString(trip_start_eva),
            service_num,
            trip_start_schedule_time);
  // clang-format on
  fbb.Finish(CreateMessage(
      fbb, 0, 0, 0, MessageUnion_TrackMessage,
      CreateTrackMessage(fbb, trip_id, fbb.CreateVector(events)).Union()));
}

msg_ptr get_track_ris_msg(std::string const& eva_num, int const service_num,
                          event_type const type, time_t const schedule_time,
                          std::string const& trip_start_eva,
                          time_t const trip_start_schedule_time,
                          std::string const& updated_track) {
  FlatBufferBuilder is_msg_fbb;
  create_track_msg(is_msg_fbb, eva_num, service_num, type, schedule_time,
                   trip_start_eva, trip_start_schedule_time, updated_track);
  message_creator mc;
  std::vector<Offset<MessageHolder>> const messages{CreateMessageHolder(
      mc,
      mc.CreateVector(is_msg_fbb.GetBufferPointer(), is_msg_fbb.GetSize()))};
  mc.create_and_finish(MsgContent_RISBatch,
                       CreateRISBatch(mc, mc.CreateVector(messages)).Union(),
                       "/ris/messages", DestinationType_Topic);
  return make_msg(mc);
}

}  // namespace motis::test::schedule::update_journey
