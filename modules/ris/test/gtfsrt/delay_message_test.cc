#include "gtest/gtest.h"

#include "motis/loader/loader.h"
#include "motis/ris/ris_message.h"
#include "motis/test/schedule/gtfs_minimal_swiss.h"

#include "./gtfsrt_test.h"

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_swiss::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_delay_test : public gtfsrt_test {
public:
  gtfsrt_delay_test() : gtfsrt_test(dataset_opt) {}
};

char const* simple_delay = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "22.TA.1-1-A-j19-1.22.R",
    "tripUpdate": {
      "trip": {
        "tripId": "22.TA.1-1-A-j19-1.22.R",
        "startTime": "01:07:00",
        "startDate": "20190627",
        "scheduleRelationship": "SCHEDULED",
        "routeId": "1-1-A-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "departure": {
          "delay": 0
        },
        "stopId": "8502113:0:1",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 5,
        "arrival": {
          "delay": -60
        },
        "departure": {
          "delay": 0
        },
        "stopId": "8502247:0:5",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 7,
        "arrival": {
          "delay": -240
        },
        "stopId": "8500309:0:5",
        "scheduleRelationship": "SCHEDULED"
      }]
    }
  }]
}
)";

constexpr auto const TIMEZONE_OFFSET = -7200;

TEST_F(gtfsrt_delay_test, simple_delay) {
  auto const msgs = parse_json(simple_delay);

  // currently only Is_ Messages and no Forecast expected
  ASSERT_EQ(1, msgs.size());

  auto const& message = msgs[0];
  EXPECT_EQ(1561596300, message.timestamp_);
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, message.earliest_);
  EXPECT_EQ(1561598940 + TIMEZONE_OFFSET, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8502113:0:1", id->station_id()->c_str());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, id->schedule_time());
  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(4, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8502113:0:1", e0->base()->station_id()->c_str());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, e0->updated_time());

  auto e1 = events->Get(1);
  EXPECT_STREQ("8502247:0:5", e1->base()->station_id()->c_str());
  EXPECT_EQ(1561598520 + TIMEZONE_OFFSET, e1->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->base()->type());
  EXPECT_EQ(1561598460 + TIMEZONE_OFFSET, e1->updated_time());

  auto e2 = events->Get(2);
  EXPECT_STREQ("8502247:0:5", e2->base()->station_id()->c_str());
  EXPECT_EQ(1561598520 + TIMEZONE_OFFSET, e2->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e2->base()->type());
  EXPECT_EQ(1561598520 + TIMEZONE_OFFSET, e2->updated_time());

  auto e3 = events->Get(3);
  EXPECT_STREQ("8500309:0:5", e3->base()->station_id()->c_str());
  EXPECT_EQ(1561598940 + TIMEZONE_OFFSET, e3->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e3->base()->type());
  EXPECT_EQ(1561598700 + TIMEZONE_OFFSET, e3->updated_time());
}

char const* simple_delay2 = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "22.TA.1-1-A-j19-1.22.R",
    "tripUpdate": {
      "trip": {
        "tripId": "22.TA.1-1-A-j19-1.22.R",
        "startTime": "01:07:00",
        "startDate": "20190627",
        "scheduleRelationship": "SCHEDULED",
        "routeId": "1-1-A-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "departure": {
          "delay": 0
        },
        "stopId": "8502113:0:1",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 5,
        "arrival": {
          "delay": -60
        },
        "departure": {
          "delay": 180
        },
        "stopId": "8502247:0:5",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 7,
        "arrival": {
          "delay": 300
        },
        "stopId": "8500309:0:5",
        "scheduleRelationship": "SCHEDULED"
      }]
    }
  }]
}
)";

TEST_F(gtfsrt_delay_test, simple_delay2) {
  auto const msgs = parse_json(simple_delay2);

  // currently only Is_ Messages and no Forecast expected
  ASSERT_EQ(1, msgs.size());

  auto const& message = msgs[0];
  EXPECT_EQ(1561596300, message.timestamp_);
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, message.earliest_);
  EXPECT_EQ(1561599240 + TIMEZONE_OFFSET, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8502113:0:1", id->station_id()->c_str());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, id->schedule_time());
  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(4, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8502113:0:1", e0->base()->station_id()->c_str());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());
  EXPECT_EQ(1561597620 + TIMEZONE_OFFSET, e0->updated_time());

  auto e1 = events->Get(1);
  EXPECT_STREQ("8502247:0:5", e1->base()->station_id()->c_str());
  EXPECT_EQ(1561598520 + TIMEZONE_OFFSET, e1->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->base()->type());
  EXPECT_EQ(1561598460 + TIMEZONE_OFFSET, e1->updated_time());

  auto e2 = events->Get(2);
  EXPECT_STREQ("8502247:0:5", e2->base()->station_id()->c_str());
  EXPECT_EQ(1561598520 + TIMEZONE_OFFSET, e2->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e2->base()->type());
  EXPECT_EQ(1561598700 + TIMEZONE_OFFSET, e2->updated_time());

  auto e3 = events->Get(3);
  EXPECT_STREQ("8500309:0:5", e3->base()->station_id()->c_str());
  EXPECT_EQ(1561598940 + TIMEZONE_OFFSET, e3->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e3->base()->type());
  EXPECT_EQ(1561599240 + TIMEZONE_OFFSET, e3->updated_time());
}

}  // namespace motis::ris::gtfsrt