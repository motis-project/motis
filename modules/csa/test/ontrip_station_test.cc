#include "gtest/gtest.h"

#include <string_view>
#include <tuple>

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
using motis::test::schedule::simple_realtime::dataset_opt_short;

constexpr auto const SEARCH_TYPE = 0;
constexpr auto const TARGET = 1;

struct csa_ontrip_station
    : public motis_instance_test,
      public ::testing::WithParamInterface<
          std::tuple<motis::routing::SearchType, char const*>> {
  csa_ontrip_station()
      : motis::test::motis_instance_test(dataset_opt_short, {"csa"}) {}
  csa_ontrip_station(csa_ontrip_station const&) = delete;
  csa_ontrip_station(csa_ontrip_station&&) = delete;
  csa_ontrip_station& operator=(csa_ontrip_station const&) = delete;
  csa_ontrip_station& operator=(csa_ontrip_station&&) = delete;
  ~csa_ontrip_station() override = default;
};

TEST_P(csa_ontrip_station, simple_fwd) {  // NOLINT
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
          std::get<SEARCH_TYPE>(GetParam()), SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      std::get<TARGET>(GetParam()));
  auto const msg = call(make_msg(fbb));
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

TEST_P(csa_ontrip_station, simple_bwd) {  // NOLINT
  if (std::string_view{"/csa/gpu"} == std::get<TARGET>(GetParam())) {
    return;  // not yet implemented
  }
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
          std::get<SEARCH_TYPE>(GetParam()), SearchDir_Backward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      std::get<TARGET>(GetParam()));
  auto const msg = call(make_msg(fbb));
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

TEST_P(csa_ontrip_station, interchange_fwd) {  // NOLINT
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000068"),
                                 fbb.CreateString("")),
              unix_time(1400))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000207"),
                             fbb.CreateString("")),
          std::get<SEARCH_TYPE>(GetParam()), SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      std::get<TARGET>(GetParam()));
  auto const msg = call(make_msg(fbb));
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);
}

TEST_P(csa_ontrip_station, interchange_fwd_pretrip) {  // NOLINT
  auto const interval = Interval{unix_time(1400), unix_time(1500)};
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("8000068"),
                                 fbb.CreateString("")),
              &interval)
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("8000207"),
                             fbb.CreateString("")),
          std::get<SEARCH_TYPE>(GetParam()), SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      std::get<TARGET>(GetParam()));
  auto const msg = call(make_msg(fbb));

  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  int i = 0;
  EXPECT_EQ("8000068", j.stops_[i++].eva_no_);
  EXPECT_EQ("8000105", j.stops_[i++].eva_no_);
  EXPECT_EQ("8070003", j.stops_[i++].eva_no_);
  EXPECT_EQ("8073368", j.stops_[i++].eva_no_);
  EXPECT_EQ("8003368", j.stops_[i++].eva_no_);
  EXPECT_EQ("8000207", j.stops_[i++].eva_no_);
}

#ifdef MOTIS_CUDA
INSTANTIATE_TEST_SUITE_P(
    csa_ontrip_station, csa_ontrip_station,
    ::testing::Values(std::make_tuple(SearchType_Default, "/csa/cpu"),
                      std::make_tuple(SearchType_Default, "/csa/cpu/sse"),
                      std::make_tuple(SearchType_Default, "/csa/gpu")));
#else
INSTANTIATE_TEST_SUITE_P(
    csa_ontrip_station, csa_ontrip_station,
    ::testing::Values(std::make_tuple(SearchType_Default, "/csa/cpu"),
                      std::make_tuple(SearchType_Default, "/csa/cpu/sse")));
#endif