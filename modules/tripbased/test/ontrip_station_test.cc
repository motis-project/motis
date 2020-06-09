#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;
using motis::test::schedule::simple_realtime::dataset_opt;

struct tripbased_ontrip_station : public motis_instance_test {
  tripbased_ontrip_station()
      : motis::test::motis_instance_test(dataset_opt, {"tripbased"},
                                         {"--tripbased.use_data_file=false"}) {}
};

TEST_F(tripbased_ontrip_station, simple_fwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000031"),
                                 fbb.CreateString("")),
              unix_time(1400))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000105"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(3, j.stops_.size());
  ASSERT_EQ(1, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("8000031", s0.eva_no_);
  EXPECT_EQ(unix_time(1409), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000068", s1.eva_no_);
  EXPECT_EQ(unix_time(1422), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000105", s2.eva_no_);
  EXPECT_EQ(unix_time(1440), s2.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_EQ("IC", m0.category_name_);
  EXPECT_EQ(2292, m0.train_nr_);
}

TEST_F(tripbased_ontrip_station, simple_bwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("")),
              unix_time(1445))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000031"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Backward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(3, j.stops_.size());
  ASSERT_EQ(1, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("8000031", s0.eva_no_);
  EXPECT_EQ(unix_time(1409), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000068", s1.eva_no_);
  EXPECT_EQ(unix_time(1422), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000105", s2.eva_no_);
  EXPECT_EQ(unix_time(1440), s2.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_EQ("IC", m0.category_name_);
  EXPECT_EQ(2292, m0.train_nr_);
}

TEST_F(tripbased_ontrip_station, intermodal_start_fwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("")),
              unix_time(1355))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000105"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("8000031"), 5, 0, 0, 1234)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(4, j.stops_.size());
  ASSERT_EQ(2, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("START", s0.eva_no_);
  EXPECT_EQ(unix_time(1359), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000031", s1.eva_no_);
  EXPECT_EQ(unix_time(1404), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1409), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000068", s2.eva_no_);
  EXPECT_EQ(unix_time(1422), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("8000105", s3.eva_no_);
  EXPECT_EQ(unix_time(1440), s3.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_TRUE(m0.is_walk_);
  EXPECT_EQ(1234, m0.mumo_id_);

  auto const& m1 = j.transports_[1];
  EXPECT_FALSE(m1.is_walk_);
  EXPECT_EQ("IC", m1.category_name_);
  EXPECT_EQ(2292, m1.train_nr_);
}

TEST_F(tripbased_ontrip_station, intermodal_start_bwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("")),
              unix_time(1445))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("START"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Backward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("8000031"), 5, 0, 0, 1234)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(4, j.stops_.size());
  ASSERT_EQ(2, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("START", s0.eva_no_);
  EXPECT_EQ(unix_time(1359), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000031", s1.eva_no_);
  EXPECT_EQ(unix_time(1404), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1409), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000068", s2.eva_no_);
  EXPECT_EQ(unix_time(1422), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("8000105", s3.eva_no_);
  EXPECT_EQ(unix_time(1440), s3.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_TRUE(m0.is_walk_);
  EXPECT_EQ(1234, m0.mumo_id_);

  auto const& m1 = j.transports_[1];
  EXPECT_FALSE(m1.is_walk_);
  EXPECT_EQ("IC", m1.category_name_);
  EXPECT_EQ(2292, m1.train_nr_);
}

TEST_F(tripbased_ontrip_station, intermodal_destination_fwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000031"),
                                 fbb.CreateString("")),
              unix_time(1355))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("END"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("END"), 5, 0, 0, 5678)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(4, j.stops_.size());
  ASSERT_EQ(2, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("8000031", s0.eva_no_);
  EXPECT_EQ(unix_time(1409), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000068", s1.eva_no_);
  EXPECT_EQ(unix_time(1422), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000105", s2.eva_no_);
  EXPECT_EQ(unix_time(1440), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1445), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("END", s3.eva_no_);
  EXPECT_EQ(unix_time(1450), s3.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_FALSE(m0.is_walk_);
  EXPECT_EQ("IC", m0.category_name_);
  EXPECT_EQ(2292, m0.train_nr_);

  auto const& m1 = j.transports_[1];
  EXPECT_TRUE(m1.is_walk_);
  EXPECT_EQ(5678, m1.mumo_id_);
}

TEST_F(tripbased_ontrip_station, intermodal_destination_bwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("END"),
                                 fbb.CreateString("")),
              unix_time(1500))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000031"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Backward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("END"), 5, 0, 0, 5678)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(4, j.stops_.size());
  ASSERT_EQ(2, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("8000031", s0.eva_no_);
  EXPECT_EQ(unix_time(1409), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000068", s1.eva_no_);
  EXPECT_EQ(unix_time(1422), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000105", s2.eva_no_);
  EXPECT_EQ(unix_time(1440), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1445), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("END", s3.eva_no_);
  EXPECT_EQ(unix_time(1450), s3.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_FALSE(m0.is_walk_);
  EXPECT_EQ("IC", m0.category_name_);
  EXPECT_EQ(2292, m0.train_nr_);

  auto const& m1 = j.transports_[1];
  EXPECT_TRUE(m1.is_walk_);
  EXPECT_EQ(5678, m1.mumo_id_);
}

TEST_F(tripbased_ontrip_station, intermodal_start_and_destination_fwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("")),
              unix_time(1355))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("END"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("8000031"), 5, 0, 0, 1234)
                      .Union()),
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("END"), 5, 0, 0, 5678)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(5, j.stops_.size());
  ASSERT_EQ(3, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("START", s0.eva_no_);
  EXPECT_EQ(unix_time(1359), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000031", s1.eva_no_);
  EXPECT_EQ(unix_time(1404), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1409), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000068", s2.eva_no_);
  EXPECT_EQ(unix_time(1422), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("8000105", s3.eva_no_);
  EXPECT_EQ(unix_time(1440), s3.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1445), s3.departure_.timestamp_);

  auto const& s4 = j.stops_[4];
  EXPECT_EQ("END", s4.eva_no_);
  EXPECT_EQ(unix_time(1450), s4.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_TRUE(m0.is_walk_);
  EXPECT_EQ(1234, m0.mumo_id_);

  auto const& m1 = j.transports_[1];
  EXPECT_FALSE(m1.is_walk_);
  EXPECT_EQ("IC", m1.category_name_);
  EXPECT_EQ(2292, m1.train_nr_);

  auto const& m2 = j.transports_[2];
  EXPECT_TRUE(m2.is_walk_);
  EXPECT_EQ(5678, m2.mumo_id_);
}

TEST_F(tripbased_ontrip_station, intermodal_start_and_destination_bwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("END"),
                                 fbb.CreateString("")),
              unix_time(1508))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("START"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Backward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>{
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("START"),
                                 fbb.CreateString("8000031"), 5, 0, 0, 1234)
                      .Union()),
              CreateAdditionalEdgeWrapper(
                  fbb, AdditionalEdge_MumoEdge,
                  CreateMumoEdge(fbb, fbb.CreateString("8000105"),
                                 fbb.CreateString("END"), 5, 0, 0, 5678)
                      .Union())}))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(5, j.stops_.size());
  ASSERT_EQ(3, j.transports_.size());
  ASSERT_EQ(1, j.trips_.size());

  auto const& s0 = j.stops_[0];
  EXPECT_EQ("START", s0.eva_no_);
  EXPECT_EQ(unix_time(1359), s0.departure_.timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000031", s1.eva_no_);
  EXPECT_EQ(unix_time(1404), s1.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1409), s1.departure_.timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000068", s2.eva_no_);
  EXPECT_EQ(unix_time(1422), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1424), s2.departure_.timestamp_);

  auto const& s3 = j.stops_[3];
  EXPECT_EQ("8000105", s3.eva_no_);
  EXPECT_EQ(unix_time(1440), s3.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1445), s3.departure_.timestamp_);

  auto const& s4 = j.stops_[4];
  EXPECT_EQ("END", s4.eva_no_);
  EXPECT_EQ(unix_time(1450), s4.arrival_.timestamp_);

  auto const& m0 = j.transports_[0];
  EXPECT_TRUE(m0.is_walk_);
  EXPECT_EQ(1234, m0.mumo_id_);

  auto const& m1 = j.transports_[1];
  EXPECT_FALSE(m1.is_walk_);
  EXPECT_EQ("IC", m1.category_name_);
  EXPECT_EQ(2292, m1.train_nr_);

  auto const& m2 = j.transports_[2];
  EXPECT_TRUE(m2.is_walk_);
  EXPECT_EQ(5678, m2.mumo_id_);
}