#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "motis/core/access/station_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/platform_interchange.h"

using namespace flatbuffers;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using motis::test::schedule::platform_interchange::dataset_opt;

namespace motis::routing {

struct platform_interchange_rt_test_base : public motis_instance_test {
  explicit platform_interchange_rt_test_base(
      std::vector<std::string> const& modules_cmdline_opt)
      : motis::test::motis_instance_test(dataset_opt, {"routing", "ris", "rt"},
                                         modules_cmdline_opt) {}

  msg_ptr routing_request() const {
    auto const interval = Interval(unix_time(1000), unix_time(1200));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("0000002"),
                                   fbb.CreateString("")),
                &interval, 0, false, false)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("0000009"),
                               fbb.CreateString("")),
            SearchType_Default, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        "/routing");
    return make_msg(fbb);
  }
};

struct platform_interchange_rt1_test
    : public platform_interchange_rt_test_base {
  platform_interchange_rt1_test()
      : platform_interchange_rt_test_base(
            {"--ris.input=test/schedule/platform_interchange/risml/"
             "track1.xml",
             "--ris.init_time=2015-11-24T10:00:00"}) {}
};

TEST_F(platform_interchange_rt1_test, track_change1) {
  auto res = call(routing_request());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];

  ASSERT_EQ(4, j.stops_.size());
  auto const& s0 = j.stops_[0];
  EXPECT_EQ("0000002", s0.eva_no_);
  EXPECT_EQ(unix_time(1110), s0.departure_.schedule_timestamp_);
  auto const& s1 = j.stops_[1];
  EXPECT_EQ("0000003", s1.eva_no_);
  EXPECT_EQ(unix_time(1200), s1.arrival_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1210), s1.departure_.schedule_timestamp_);
  EXPECT_EQ("1", s1.arrival_.schedule_track_);
  EXPECT_EQ("3", s1.arrival_.track_);
  EXPECT_EQ("4", s1.departure_.schedule_track_);
  EXPECT_EQ("4", s1.departure_.track_);
  auto const& s2 = j.stops_[2];
  EXPECT_EQ("0000008", s2.eva_no_);
  EXPECT_EQ(unix_time(1330), s2.arrival_.schedule_timestamp_);
  EXPECT_EQ(unix_time(1335), s2.departure_.schedule_timestamp_);
  auto const& s3 = j.stops_[3];
  EXPECT_EQ("0000009", s3.eva_no_);
  EXPECT_EQ(unix_time(1425), s3.arrival_.schedule_timestamp_);

  ASSERT_EQ(2, j.transports_.size());
  auto const& t0 = j.transports_[0];
  EXPECT_EQ(1, t0.train_nr_);
  auto const& t1 = j.transports_[1];
  EXPECT_EQ(3, t1.train_nr_);
}

struct platform_interchange_rt2_test
    : public platform_interchange_rt_test_base {
  platform_interchange_rt2_test()
      : platform_interchange_rt_test_base(
            {"--ris.input=test/schedule/platform_interchange/risml/",
             "--ris.init_time=2015-11-24T10:01:00"}) {}
};

TEST_F(platform_interchange_rt2_test, track_change1) {
  auto res = call(routing_request());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(2, journeys.size());

  {
    auto const& j = journeys[0];
    ASSERT_EQ(4, j.stops_.size());
    auto const& s0 = j.stops_[0];
    EXPECT_EQ("0000002", s0.eva_no_);
    EXPECT_EQ(unix_time(1108), s0.departure_.schedule_timestamp_);
    auto const& s1 = j.stops_[1];
    EXPECT_EQ("0000003", s1.eva_no_);
    EXPECT_EQ(unix_time(1200), s1.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1204), s1.departure_.schedule_timestamp_);
    EXPECT_EQ("3", s1.arrival_.schedule_track_);
    EXPECT_EQ("1", s1.arrival_.track_);
    EXPECT_EQ("2", s1.departure_.schedule_track_);
    EXPECT_EQ("2", s1.departure_.track_);
    auto const& s2 = j.stops_[2];
    EXPECT_EQ("0000008", s2.eva_no_);
    EXPECT_EQ(unix_time(1320), s2.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1325), s2.departure_.schedule_timestamp_);
    auto const& s3 = j.stops_[3];
    EXPECT_EQ("0000009", s3.eva_no_);
    EXPECT_EQ(unix_time(1420), s3.arrival_.schedule_timestamp_);

    ASSERT_EQ(2, j.transports_.size());
    auto const& t0 = j.transports_[0];
    EXPECT_EQ(4, t0.train_nr_);
    auto const& t1 = j.transports_[1];
    EXPECT_EQ(2, t1.train_nr_);
  }

  {
    auto const& j = journeys[1];
    ASSERT_EQ(4, j.stops_.size());
    auto const& s0 = j.stops_[0];
    EXPECT_EQ("0000002", s0.eva_no_);
    EXPECT_EQ(unix_time(1110), s0.departure_.schedule_timestamp_);
    auto const& s1 = j.stops_[1];
    EXPECT_EQ("0000003", s1.eva_no_);
    EXPECT_EQ(unix_time(1200), s1.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1210), s1.departure_.schedule_timestamp_);
    EXPECT_EQ("1", s1.arrival_.schedule_track_);
    EXPECT_EQ("3", s1.arrival_.track_);
    EXPECT_EQ("4", s1.departure_.schedule_track_);
    EXPECT_EQ("4", s1.departure_.track_);
    auto const& s2 = j.stops_[2];
    EXPECT_EQ("0000008", s2.eva_no_);
    EXPECT_EQ(unix_time(1330), s2.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1335), s2.departure_.schedule_timestamp_);
    auto const& s3 = j.stops_[3];
    EXPECT_EQ("0000009", s3.eva_no_);
    EXPECT_EQ(unix_time(1425), s3.arrival_.schedule_timestamp_);

    ASSERT_EQ(2, j.transports_.size());
    auto const& t0 = j.transports_[0];
    EXPECT_EQ(1, t0.train_nr_);
    auto const& t1 = j.transports_[1];
    EXPECT_EQ(3, t1.train_nr_);
  }
}

}  // namespace motis::routing
