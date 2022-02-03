#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::routing {

struct routing_rt : public motis_instance_test {
  routing_rt()
      : motis::test::motis_instance_test(
            dataset_opt, {"routing", "ris", "rt"},
            {"--ris.input=test/schedule/simple_realtime/risml/delays.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}

  msg_ptr routing_request() const {
    auto const interval = Interval(unix_time(1355), unix_time(1355));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("8000260"),
                                   fbb.CreateString("")),
                &interval)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("8000208"),
                               fbb.CreateString("")),
            SearchType_SingleCriterion, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        "/routing");
    return make_msg(fbb);
  }
};

TEST_F(routing_rt, finds_annotated_connections) {
  auto res = call(routing_request());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(1, journeys.size());
  auto j = journeys[0];

  // ICE 628
  auto s0 = j.stops_[0];  // Wuerzburg
  EXPECT_EQ("8000260", s0.eva_no_);
  EXPECT_EQ(0, s0.arrival_.schedule_timestamp_);
  EXPECT_EQ(0, s0.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1355), s0.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1355), s0.departure_.timestamp_);

  auto s1 = j.stops_[1];  // Aschaffenburg
  EXPECT_EQ(unix_time(1436), s1.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1437), s1.departure_.timestamp_);

  auto s2 = j.stops_[2];  // Frankfurt(Main)Hbf
  EXPECT_EQ(unix_time(1504), s2.arrival_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1505), s2.arrival_.timestamp_);
  EXPECT_EQ(unix_time(1510), s2.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1510), s2.departure_.timestamp_);

  auto s3 = j.stops_[3];  // Frankfurt(M) Flughafe
  EXPECT_EQ(unix_time(1525), s3.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1530), s3.departure_.timestamp_);

  // walk
  auto s4 = j.stops_[4];  // Koeln Messe/Deutz Gl.1
  EXPECT_EQ(unix_time(1614), s4.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1619), s4.departure_.timestamp_);

  auto s5 = j.stops_[5];  // Koeln Messe/Deutz
  EXPECT_EQ(unix_time(1615), s5.arrival_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1620), s5.arrival_.timestamp_);

  // RE 10958
  EXPECT_EQ(unix_time(1633), s5.departure_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1633), s5.departure_.timestamp_);

  auto s7 = j.stops_[7];  // Koeln-Ehrenfeld
  EXPECT_EQ("8000208", s7.eva_no_);
  EXPECT_EQ(unix_time(1651), s7.arrival_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1651), s7.arrival_.timestamp_);
}

}  // namespace motis::routing
