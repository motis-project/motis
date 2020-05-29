#include "gtest/gtest.h"

#include "motis/loader/loader.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/test/schedule/gtfs_minimal_swiss.h"

#include "./gtfsrt_test.h"

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_swiss::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_cancel_test : public gtfsrt_test {
public:
  gtfsrt_cancel_test() : gtfsrt_test(dataset_opt) {}
};

char const* simple_cancel = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "7.TA.1-1-A-j19-1.7.H",
    "tripUpdate": {
      "trip": {
        "tripId": "7.TA.1-1-A-j19-1.7.H",
        "startTime": "01:00:00",
        "startDate": "20190627",
        "scheduleRelationship": "CANCELED",
        "routeId": "1-1-A-j19-1"
      }
    }
  }]
}
)";

constexpr auto const TIMEZONE_OFFSET = -7200;

TEST_F(gtfsrt_cancel_test, simple_cancel) {
  auto const msgs = parse_json(simple_cancel);

  ASSERT_EQ(1, msgs.size());

  auto const& message = msgs[0];
  EXPECT_EQ(1561596300, message.timestamp_);
  EXPECT_EQ(1561597200 + TIMEZONE_OFFSET, message.earliest_);
  EXPECT_EQ(1561600800 + TIMEZONE_OFFSET, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_CancelMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<CancelMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8503000:0:41/42", id->station_id()->c_str());
  EXPECT_EQ(1561597200 + TIMEZONE_OFFSET, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(34, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8503000:0:41/42", e0->station_id()->c_str());
  EXPECT_EQ(EventType_DEP, e0->type());
  EXPECT_EQ(1561597200 + TIMEZONE_OFFSET, e0->schedule_time());

  auto e1 = events->Get(1);
  EXPECT_STREQ("8503020:0:4", e1->station_id()->c_str());
  EXPECT_EQ(EventType_ARR, e1->type());
  EXPECT_EQ(1561597320 + TIMEZONE_OFFSET, e1->schedule_time());

  auto e2 = events->Get(2);
  EXPECT_STREQ("8503020:0:4", e2->station_id()->c_str());
  EXPECT_EQ(EventType_DEP, e2->type());
  EXPECT_EQ(1561597320 + TIMEZONE_OFFSET, e2->schedule_time());

  auto e32 = events->Get(32);
  EXPECT_STREQ("8502114:0:1", e32->station_id()->c_str());
  EXPECT_EQ(EventType_DEP, e32->type());
  EXPECT_EQ(1561600500 + TIMEZONE_OFFSET, e32->schedule_time());

  auto e33 = events->Get(33);
  EXPECT_STREQ("8502113:0:4", e33->station_id()->c_str());
  EXPECT_EQ(EventType_ARR, e33->type());
  EXPECT_EQ(1561600800 + TIMEZONE_OFFSET, e33->schedule_time());
}

TEST_F(gtfsrt_cancel_test, cancel_twice) {
  std::string json = simple_cancel;
  auto bin = json_to_protobuf(json);
  auto view = std::string_view{bin.c_str(), bin.size()};
  gtfsrt_parser cut;
  auto const msgs1 = cut.parse(*sched_, view);

  ASSERT_EQ(1, msgs1.size());

  auto const msgs2 = cut.parse(*sched_, view);
  ASSERT_EQ(0, msgs2.size());
}

}  // namespace motis::ris::gtfsrt