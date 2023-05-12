#include "gtest/gtest.h"

#include "utl/to_vec.h"

#include "motis/core/conv/event_type_conv.h"

#include "motis/test/motis_instance_test.h"

#include "./schedule.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::routing;
using motis::cc::dataset_opt;

struct id_event {
  id_event(std::string station_id, int train_num, int schedule_hhmm)
      : station_id_(std::move(station_id)),
        train_num_(train_num),
        schedule_hhmm_(schedule_hhmm) {}

  Offset<ris::IdEvent> to_fbs(schedule const& sched,
                              FlatBufferBuilder& fbb) const {
    return ris::CreateIdEvent(fbb, fbb.CreateString(station_id_), train_num_,
                              unix_time(sched, schedule_hhmm_));
  }

  std::string station_id_;
  int train_num_;
  int schedule_hhmm_;
};

struct event {
  event(std::string station_id, int train_num, std::string line_id,
        event_type ev_type, int schedule_time_hhmm)
      : station_id_(std::move(station_id)),
        train_num_(train_num),
        line_id_(std::move(line_id)),
        ev_type_(ev_type),
        schedule_time_hhmm_(schedule_time_hhmm) {}

  Offset<ris::Event> to_fbs(schedule const& sched,
                            FlatBufferBuilder& fbb) const {
    return ris::CreateEvent(fbb, fbb.CreateString(station_id_), train_num_,
                            fbb.CreateString(line_id_), motis::to_fbs(ev_type_),
                            unix_time(sched, schedule_time_hhmm_));
  }

  std::string station_id_;
  int train_num_;
  std::string line_id_;
  event_type ev_type_;
  int schedule_time_hhmm_;
};

struct updated_event {
  updated_event(event ev, int new_time_hhmm)
      : ev_(std::move(ev)), new_time_hhmm_(new_time_hhmm) {}

  Offset<ris::UpdatedEvent> to_fbs(schedule const& sched,
                                   FlatBufferBuilder& fbb) const {
    return ris::CreateUpdatedEvent(fbb, ev_.to_fbs(sched, fbb),
                                   unix_time(sched, new_time_hhmm_));
  }

  event ev_;
  int new_time_hhmm_;
};

struct rerouted_event {
  rerouted_event(event ev, std::string category, std::string track,
                 bool withdrawal)
      : ev_(std::move(ev)),
        category_(std::move(category)),
        track_(std::move(track)),
        withdrawal_(withdrawal) {}

  Offset<ris::ReroutedEvent> to_fbs(schedule const& sched,
                                    FlatBufferBuilder& fbb) const {
    return ris::CreateReroutedEvent(
        fbb,
        ris::CreateAdditionalEvent(fbb, ev_.to_fbs(sched, fbb),
                                   fbb.CreateString(category_),
                                   fbb.CreateString(track_)),
        withdrawal_ ? ris::RerouteStatus_Normal : ris::RerouteStatus_UmlNeu);
  }

  event ev_;
  std::string category_;
  std::string track_;
  bool withdrawal_;
};

struct cc_check_routed_connection_test : public motis_instance_test {
  cc_check_routed_connection_test()
      : motis::test::motis_instance_test(dataset_opt, {"cc", "routing", "rt"}) {
  }

  msg_ptr route(std::string const& from, std::string const& to,
                int hhmm) const {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_OntripStationStart,
            CreateOntripStationStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString(from),
                                   fbb.CreateString("")),
                unix_time(hhmm))
                .Union(),
            CreateInputStation(fbb, fbb.CreateString(to), fbb.CreateString("")),
            SearchType_SingleCriterion, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        "/routing");
    return make_msg(fbb);
  }

  static msg_ptr check_connection(Connection const* con) {
    message_creator fbb;
    fbb.create_and_finish(MsgContent_Connection,
                          motis_copy_table(Connection, fbb, con).Union(),
                          "/cc");
    return make_msg(fbb);
  }

  static msg_ptr reroute(schedule const& sched, id_event const& id,
                         std::vector<event> const& cancel_events,
                         std::vector<rerouted_event> const& rerouted_events) {
    FlatBufferBuilder fbb;
    fbb.Finish(ris::CreateRISMessage(
        fbb, 0U, 0U, 0U, ris::RISMessageUnion_RerouteMessage,
        ris::CreateRerouteMessage(
            fbb, id.to_fbs(sched, fbb),
            fbb.CreateVector(utl::to_vec(
                cancel_events,
                [&](auto const& ev) { return ev.to_fbs(sched, fbb); })),
            fbb.CreateVector(utl::to_vec(
                rerouted_events,
                [&](auto const& ev) { return ev.to_fbs(sched, fbb); })))
            .Union()));

    message_creator mc;
    auto const msg_holders =
        std::vector<Offset<ris::RISMessageHolder>>{ris::CreateRISMessageHolder(
            mc, mc.CreateVector(fbb.GetBufferPointer(), fbb.GetSize()))};
    mc.create_and_finish(
        MsgContent_RISBatch,
        ris::CreateRISBatch(mc, mc.CreateVector(msg_holders)).Union(),
        "/ris/messages", DestinationType_Topic);

    return make_msg(mc);
  }

  static msg_ptr delay(schedule const& sched, id_event const& id,
                       std::vector<updated_event> const& delays,
                       bool is_message) {
    FlatBufferBuilder fbb;
    fbb.Finish(ris::CreateRISMessage(
        fbb, 0U, 0U, 0U, ris::RISMessageUnion_DelayMessage,
        ris::CreateDelayMessage(
            fbb, id.to_fbs(sched, fbb),
            is_message ? ris::DelayType_Is : ris::DelayType_Forecast,
            fbb.CreateVector(utl::to_vec(
                delays, [&](auto const& ev) { return ev.to_fbs(sched, fbb); })))
            .Union()));

    message_creator mc;
    auto const msg_holders =
        std::vector<Offset<ris::RISMessageHolder>>{ris::CreateRISMessageHolder(
            mc, mc.CreateVector(fbb.GetBufferPointer(), fbb.GetSize()))};
    mc.create_and_finish(
        MsgContent_RISBatch,
        ris::CreateRISBatch(mc, mc.CreateVector(msg_holders)).Union(),
        "/ris/messages", DestinationType_Topic);

    return make_msg(mc);
  }
};

TEST_F(cc_check_routed_connection_test, simple_result_ok) {
  auto check_schedule_okay = [&](std::string const& from, std::string const& to,
                                 int hhmm) {
    auto const routing_res = call(route(from, to, hhmm));
    auto const connections =
        motis_content(RoutingResponse, routing_res)->connections();
    ASSERT_EQ(1, connections->size());

    auto const cc_res = call(check_connection(connections->Get(0)));
    EXPECT_NE(MsgContent_MotisError, cc_res->get()->content_type());
  };

  check_schedule_okay("0000002", "0000011", 1700);
  check_schedule_okay("0000001", "0000005", 1700);
  check_schedule_okay("0000003", "0000005", 1700);
  check_schedule_okay("0000003", "0000008", 1700);
}

TEST_F(cc_check_routed_connection_test, first_enter_cancelled) {
  auto const routing_res = call(route("0000003", "0000005", 1700));
  publish(reroute(sched(), id_event{"0000003", 6, 1810},
                  {event{"0000003", 6, "", event_type::DEP, 1810},
                   event{"0000004", 6, "", event_type::ARR, 1900}},
                  {}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(
      reroute(sched(), id_event{"0000003", 6, 1810}, {},
              {rerouted_event{event{"0000003", 6, "", event_type::DEP, 1810},
                              "IC", "", true},
               rerouted_event{event{"0000004", 6, "", event_type::ARR, 1900},
                              "IC", "", true}}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}

TEST_F(cc_check_routed_connection_test, last_exit_cancelled) {
  auto const routing_res = call(route("0000003", "0000005", 1700));
  publish(reroute(sched(), id_event{"0000003", 6, 1810},
                  {event{"0000004", 6, "", event_type::DEP, 1910},
                   event{"0000005", 6, "", event_type::ARR, 2000}},
                  {}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(
      reroute(sched(), id_event{"0000003", 6, 1810}, {},
              {rerouted_event{event{"0000004", 6, "", event_type::DEP, 1910},
                              "IC", "", true},
               rerouted_event{event{"0000005", 6, "", event_type::ARR, 2000},
                              "IC", "", true}}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}

TEST_F(cc_check_routed_connection_test, interchange_arrival_cancelled) {
  auto const routing_res = call(route("0000003", "0000012", 1700));
  publish(reroute(sched(), id_event{"0000003", 6, 1810},
                  {event{"0000004", 6, "", event_type::DEP, 1910},
                   event{"0000005", 6, "", event_type::ARR, 2000}},
                  {}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(
      reroute(sched(), id_event{"0000003", 6, 1810}, {},
              {rerouted_event{event{"0000004", 6, "", event_type::DEP, 1910},
                              "IC", "", true},
               rerouted_event{event{"0000005", 6, "", event_type::ARR, 2000},
                              "IC", "", true}}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}

TEST_F(cc_check_routed_connection_test, interchange_departure_cancelled) {
  auto const routing_res = call(route("0000003", "0000012", 1700));
  publish(
      reroute(sched(), id_event{"0000005", 9, 2010},
              {event{"0000005", 9, "", event_type::DEP, 2010},
               event{"0000012", 9, "", event_type::ARR, 2100}},
              {rerouted_event{event{"0000012", 9, "", event_type::DEP, 2110},
                              "IC", "", false},
               rerouted_event{event{"0000011", 9, "", event_type::ARR, 2200},
                              "IC", "", false}}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(
      reroute(sched(), id_event{"0000005", 9, 2010}, {},
              {rerouted_event{event{"0000005", 9, "", event_type::DEP, 2010},
                              "IC", "", true},
               rerouted_event{event{"0000012", 9, "", event_type::ARR, 2100},
                              "IC", "", true}}));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}

TEST_F(cc_check_routed_connection_test, interchange_arrival_delay) {
  auto const routing_res = call(route("0000003", "0000012", 1700));
  publish(delay(
      sched(), id_event{"0000003", 6, 1810},
      {updated_event{event{"0000005", 6, "", event_type::ARR, 2000}, 2015}},
      false));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(delay(
      sched(), id_event{"0000003", 6, 1810},
      {updated_event{event{"0000005", 6, "", event_type::ARR, 2000}, 2001}},
      true));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}

TEST_F(cc_check_routed_connection_test, walk_interchange_arrival_delay) {
  auto const routing_res = call(route("0000003", "0000009", 1700));
  publish(delay(
      sched(), id_event{"0000003", 6, 1810},
      {updated_event{event{"0000005", 6, "", event_type::ARR, 2000}, 2051}},
      false));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_THROW(  // NO_LINT
      call(check_connection(
          motis_content(RoutingResponse, routing_res)->connections()->Get(0))),
      std::runtime_error);

  publish(delay(
      sched(), id_event{"0000003", 6, 1810},
      {updated_event{event{"0000005", 6, "", event_type::ARR, 2000}, 2001}},
      true));
  publish(make_no_msg("/ris/system_time_changed"));
  EXPECT_NO_THROW(call(check_connection(  // NO_LINT
      motis_content(RoutingResponse, routing_res)->connections()->Get(0))));
}
