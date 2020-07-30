#include "gtest/gtest.h"

#include <algorithm>

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/test/motis_instance_test.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;

struct tripbased_pretrip : public motis_instance_test {
  tripbased_pretrip()
      : motis::test::motis_instance_test(
            loader::loader_options{
                {"modules/tripbased/test_resources/schedule"}, "20151121"},
            {"tripbased"}, {"--tripbased.use_data_file=false"}) {}

  bool has_journey(std::vector<journey> const& journeys, int const departure,
                   int const arrival) {
    return std::any_of(begin(journeys), end(journeys), [&](journey const& j) {
      auto const dep = unix_time(departure);
      auto const arr = unix_time(arrival);
      return !j.stops_.empty() &&
             j.stops_.front().departure_.schedule_timestamp_ == dep &&
             j.stops_.back().arrival_.schedule_timestamp_ == arr;
    });
  }
};

TEST_F(tripbased_pretrip, simple_fwd) {
  message_creator fbb;
  auto const interval = Interval(unix_time(1500), unix_time(1700));
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("2000001"),
                                 fbb.CreateString("")),
              &interval, 0, false, false)
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("1000001"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);
  EXPECT_EQ(4, journeys.size());
  for (auto const& j : journeys) {
    EXPECT_EQ(3, j.stops_.size());
    if (j.stops_.size() >= 3) {
      EXPECT_EQ("2000001", j.stops_[0].eva_no_);
      EXPECT_EQ("6000001", j.stops_[1].eva_no_);
      EXPECT_EQ("1000001", j.stops_[2].eva_no_);
    }
    EXPECT_EQ(1, j.transports_.size());
    if (!j.transports_.empty()) {
      EXPECT_EQ("RE", j.transports_[0].category_name_);
      EXPECT_EQ(2, j.transports_[0].train_nr_);
    }
  }
  EXPECT_TRUE(has_journey(journeys, 1512, 1555));
  EXPECT_TRUE(has_journey(journeys, 1542, 1625));
  EXPECT_TRUE(has_journey(journeys, 1612, 1655));
  EXPECT_TRUE(has_journey(journeys, 1642, 1725));
}
