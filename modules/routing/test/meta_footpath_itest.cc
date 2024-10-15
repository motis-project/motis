#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/multiple_starting_footpaths.h"

using namespace flatbuffers;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using motis::test::schedule::multiple_starting_foothpaths::dataset_opt;

namespace motis::routing {

struct meta_fp_start_label_generation : public motis_instance_test {
  meta_fp_start_label_generation()
      : motis::test::motis_instance_test(dataset_opt, {"routing"}, {}) {}

  msg_ptr routing_request() const {
    auto const interval = Interval(unix_time(1300), unix_time(1400));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("8001736"),
                                   fbb.CreateString("")),
                &interval, 0, false, false)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("8000105"),
                               fbb.CreateString("")),
            SearchType_SingleCriterion, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        "/routing");
    return make_msg(fbb);
  }

  msg_ptr routing_request_meta() const {
    auto const interval = Interval(unix_time(1300), unix_time(1400));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("8000284"),
                                   fbb.CreateString("")),
                &interval, 0, false, false)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("8000080"),
                               fbb.CreateString("")),
            SearchType_SingleCriterion, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
            true, true, true)
            .Union(),
        "/routing");
    return make_msg(fbb);
  }
  msg_ptr routing_request_fp() const {
    auto const interval = Interval(unix_time(1300), unix_time(1400));
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, Start_PretripStart,
            CreatePretripStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("8001736"),
                                   fbb.CreateString("")),
                &interval, 0, false, false)
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("8000105"),
                               fbb.CreateString("")),
            SearchType_SingleCriterion, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
            false, false, true)
            .Union(),
        "/routing");
    return make_msg(fbb);
  }
};

TEST_F(meta_fp_start_label_generation, generate_starting_meta) {
  auto res = call(routing_request_meta());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(2, j.stops_.size());

  // ICE 629
  auto const& s0 = j.stops_[0];  // Würzburg
  EXPECT_EQ("8000260", s0.eva_no_);
  EXPECT_EQ(unix_time(1355), s0.departure_.schedule_timestamp_);
  auto const& s1 = j.stops_[1];  // Dortmund
  EXPECT_EQ("8000080", s1.eva_no_);
  EXPECT_EQ(unix_time(1726), s1.arrival_.schedule_timestamp_);
}

TEST_F(meta_fp_start_label_generation, generate_starting_fps) {
  auto res = call(routing_request_fp());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(8, j.stops_.size());

  // IC 2292
  auto const& s0 = j.stops_[0];  // Stuttgart
  EXPECT_EQ("8001736", s0.eva_no_);
  EXPECT_EQ(unix_time(1322), s0.departure_.schedule_timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8001112", s1.eva_no_);
  EXPECT_EQ(unix_time(1327), s1.departure_.schedule_timestamp_);

  auto const& s2 = j.stops_[2];
  EXPECT_EQ("8000096", s2.eva_no_);
  EXPECT_EQ(unix_time(1330), s2.departure_.schedule_timestamp_);

  auto const& s7 = j.stops_[7];
  EXPECT_EQ("8000105", s7.eva_no_);
  EXPECT_EQ(unix_time(1440), s7.arrival_.schedule_timestamp_);
}

}  // namespace motis::routing
