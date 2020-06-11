#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/ris/risml/risml_parser.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/multiple_starting_footpaths.h"

using namespace flatbuffers;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using motis::test::schedule::multiple_starting_foothpaths::dataset_opt;

namespace motis::routing {

struct tripbased_meta_footpath : public motis_instance_test {
  tripbased_meta_footpath()
      : motis::test::motis_instance_test(dataset_opt, {"tripbased"},
                                         {"--tripbased.use_data_file=false"}) {}

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
            SearchType_Default, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
            true, true, true)
            .Union(),
        "/tripbased");
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
            SearchType_Default, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
            false, false, true)
            .Union(),
        "/tripbased");
    return make_msg(fbb);
  }
};

TEST_F(tripbased_meta_footpath, generate_starting_meta) {
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

TEST_F(tripbased_meta_footpath, generate_starting_fps) {
  auto res = call(routing_request_fp());
  auto journeys = message_to_journeys(motis_content(RoutingResponse, res));

  ASSERT_EQ(1, journeys.size());
  auto const& j = journeys[0];
  ASSERT_EQ(7, j.stops_.size());

  // IC 2292
  auto const& s0 = j.stops_[0];  // Stuttgart
  EXPECT_EQ("8001736", s0.eva_no_);
  EXPECT_EQ(unix_time(1322), s0.departure_.schedule_timestamp_);

  auto const& s1 = j.stops_[1];
  EXPECT_EQ("8000096", s1.eva_no_);
  EXPECT_EQ(unix_time(1330), s1.departure_.schedule_timestamp_);

  auto const& s6 = j.stops_[6];
  EXPECT_EQ("8000105", s6.eva_no_);
  EXPECT_EQ(unix_time(1440), s6.arrival_.schedule_timestamp_);
}

}  // namespace motis::routing
