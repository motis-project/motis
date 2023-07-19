#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"

#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;
using motis::test::schedule::simple_realtime::dataset_opt;

struct routing_ontrip_train : public motis_instance_test {
  routing_ontrip_train()
      : motis::test::motis_instance_test(dataset_opt, {"routing"}) {}
};

TEST_F(routing_ontrip_train, stay_in_train) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb,
              CreateTripId(fbb, fbb.CreateString(""),
                           fbb.CreateString("8000096"), 2292, unix_time(1305),
                           fbb.CreateString("8000105"), unix_time(1440),
                           fbb.CreateString("381")),
              CreateInputStation(fbb, fbb.CreateString("8000031"),
                                 fbb.CreateString("")),
              unix_time(1408))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000105"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  EXPECT_EQ(1, motis_content(RoutingResponse, call(make_msg(fbb)))
                   ->connections()
                   ->size());
}

TEST_F(routing_ontrip_train, stay_in_train_single_leg) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb,
              CreateTripId(fbb, fbb.CreateString(""),
                           fbb.CreateString("8000096"), 2292, unix_time(1305),
                           fbb.CreateString("8000105"), unix_time(1440),
                           fbb.CreateString("381")),
              CreateInputStation(fbb, fbb.CreateString("8000068"),
                                 fbb.CreateString("")),
              unix_time(1422))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000105"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  EXPECT_EQ(1, motis_content(RoutingResponse, call(make_msg(fbb)))
                   ->connections()
                   ->size());
}

TEST_F(routing_ontrip_train, stay_in_train_with_interchange) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb,
              CreateTripId(fbb, fbb.CreateString(""),
                           fbb.CreateString("8000096"), 2292, unix_time(1305),
                           fbb.CreateString("8000105"), unix_time(1440),
                           fbb.CreateString("381")),
              CreateInputStation(fbb, fbb.CreateString("8000068"),
                                 fbb.CreateString("")),
              unix_time(1422))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000001"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  EXPECT_EQ(1, motis_content(RoutingResponse, call(make_msg(fbb)))
                   ->connections()
                   ->size());
}

TEST_F(routing_ontrip_train, immediate_exit_with_interchange) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb,
              CreateTripId(fbb, fbb.CreateString(""),
                           fbb.CreateString("8000096"), 2292, unix_time(1305),
                           fbb.CreateString("8000105"), unix_time(1440),
                           fbb.CreateString("381")),
              CreateInputStation(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("")),
              unix_time(1440))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000001"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  EXPECT_EQ(1, motis_content(RoutingResponse, call(make_msg(fbb)))
                   ->connections()
                   ->size());
}

TEST_F(routing_ontrip_train, immediate_exit_with_walk) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb,
              CreateTripId(fbb, fbb.CreateString(""),
                           fbb.CreateString("8000261"), 628, unix_time(1154),
                           fbb.CreateString("8000080"), unix_time(1726),
                           fbb.CreateString("")),
              CreateInputStation(fbb, fbb.CreateString("8073368"),
                                 fbb.CreateString("")),
              unix_time(1614))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000001"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  EXPECT_EQ(1, motis_content(RoutingResponse, call(make_msg(fbb)))
                   ->connections()
                   ->size());
}