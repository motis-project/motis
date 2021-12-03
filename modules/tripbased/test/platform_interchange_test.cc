#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"

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
using motis::test::schedule::platform_interchange::
    dataset_without_platforms_opt;

namespace motis::tripbased {

struct tripbased_platform_interchange_test_base : public motis_instance_test {
  explicit tripbased_platform_interchange_test_base(
      loader::loader_options const& opt)
      : motis::test::motis_instance_test(opt, {"tripbased"},
                                         {"--tripbased.use_data_file=false"}) {}

  msg_ptr fwd_routing_request() const {
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
        "/tripbased");
    return make_msg(fbb);
  }

  msg_ptr bwd_routing_request() const {
    auto const interval = Interval(unix_time(1400), unix_time(1500));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("0000009"),
                                   fbb.CreateString("")),
                &interval, 0, false, false)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("0000002"),
                               fbb.CreateString("")),
            SearchType_Default, SearchDir_Backward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        "/tripbased");
    return make_msg(fbb);
  }
};

struct tripbased_platform_interchange_test
    : public tripbased_platform_interchange_test_base {
  tripbased_platform_interchange_test()
      : tripbased_platform_interchange_test_base(dataset_opt) {}

  void check_result(std::vector<journey> const& journeys) {
    ASSERT_EQ(1, journeys.size());
    auto const& j = journeys[0];

    ASSERT_EQ(4, j.stops_.size());
    auto const& s0 = j.stops_[0];
    EXPECT_EQ("0000002", s0.eva_no_);
    EXPECT_EQ(unix_time(1110), s0.departure_.schedule_timestamp_);
    auto const& s1 = j.stops_[1];
    EXPECT_EQ("0000003", s1.eva_no_);
    EXPECT_EQ(unix_time(1200), s1.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1204), s1.departure_.schedule_timestamp_);
    auto const& s2 = j.stops_[2];
    EXPECT_EQ("0000008", s2.eva_no_);
    EXPECT_EQ(unix_time(1320), s2.arrival_.schedule_timestamp_);
    EXPECT_EQ(unix_time(1325), s2.departure_.schedule_timestamp_);
    auto const& s3 = j.stops_[3];
    EXPECT_EQ("0000009", s3.eva_no_);
    EXPECT_EQ(unix_time(1420), s3.arrival_.schedule_timestamp_);

    ASSERT_EQ(2, j.transports_.size());
    auto const& t0 = j.transports_[0];
    EXPECT_EQ(1, t0.train_nr_);
    auto const& t1 = j.transports_[1];
    EXPECT_EQ(2, t1.train_nr_);
  }
};

TEST_F(tripbased_platform_interchange_test, transfer_time) {
  auto const st = get_station(sched(), "0000003");
  EXPECT_EQ(8, st->transfer_time_);
  EXPECT_EQ(4, st->platform_transfer_time_);
}

TEST_F(tripbased_platform_interchange_test, same_platform_fwd) {
  auto res = call(fwd_routing_request());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));
  check_result(journeys);
}

TEST_F(tripbased_platform_interchange_test, same_platform_bwd) {
  auto res = call(bwd_routing_request());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));
  check_result(journeys);
}

struct tripbased_platform_interchange_no_platforms_test
    : public tripbased_platform_interchange_test_base {
  tripbased_platform_interchange_no_platforms_test()
      : tripbased_platform_interchange_test_base(
            dataset_without_platforms_opt) {}
};

TEST_F(tripbased_platform_interchange_no_platforms_test, transfer_time) {
  auto const st = get_station(sched(), "0000003");
  EXPECT_EQ(8, st->transfer_time_);
  EXPECT_EQ(4, st->platform_transfer_time_);
}

TEST_F(tripbased_platform_interchange_no_platforms_test, same_platform) {
  auto res = call(fwd_routing_request());
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

}  // namespace motis::tripbased
