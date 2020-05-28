#include "gtest/gtest.h"

#include "motis/loader/loader.h"
#include "motis/ris/ris_message.h"
#include "motis/test/schedule/gtfs_minimal_swiss.h"

#include "./gtfsrt_test.h"

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_swiss::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_reroute_test : public gtfsrt_test {
public:
  gtfsrt_reroute_test() : gtfsrt_test(dataset_opt){};

  static void check_reroute_event(ris_message const&);
};

constexpr auto const TIMEZONE_OFFSET = -7200;

void gtfsrt_reroute_test::check_reroute_event(ris_message const& message) {
  EXPECT_EQ(1561612800, message.timestamp_);
  EXPECT_EQ(1561611720 + TIMEZONE_OFFSET, message.earliest_);
  EXPECT_EQ(1561615020 + TIMEZONE_OFFSET, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_RerouteMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<RerouteMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8501200:0:4", id->station_id()->c_str());
  EXPECT_EQ(1561611720 + TIMEZONE_OFFSET, id->schedule_time());

  auto events = inner_msg->cancelled_events();
  ASSERT_EQ(6, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8501200:0:4", e0->station_id()->c_str());
  EXPECT_EQ(1561611720 + TIMEZONE_OFFSET, e0->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->type());

  auto e1 = events->Get(1);
  EXPECT_STREQ("8501403:0:1", e1->station_id()->c_str());
  EXPECT_EQ(1561613520 + TIMEZONE_OFFSET, e1->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->type());

  auto e2 = events->Get(2);
  EXPECT_STREQ("8501403:0:1", e2->station_id()->c_str());
  EXPECT_EQ(1561613580 + TIMEZONE_OFFSET, e2->schedule_time());
  EXPECT_EQ(EventType_DEP, e2->type());

  auto e3 = events->Get(3);
  EXPECT_STREQ("8501506:0:2", e3->station_id()->c_str());
  EXPECT_EQ(1561615020 + TIMEZONE_OFFSET, e3->schedule_time());
  EXPECT_EQ(EventType_ARR, e3->type());

  // check the implicit events (no arrival a seq.no 2 and departure at seq-no 6)
  auto e4 = events->Get(4);
  EXPECT_STREQ("8501300:0:3", e4->station_id()->c_str());
  EXPECT_EQ(1561612140 + TIMEZONE_OFFSET, e4->schedule_time());
  EXPECT_EQ(EventType_ARR, e4->type());

  auto e5 = events->Get(5);
  EXPECT_STREQ("8501500:0:2", e5->station_id()->c_str());
  EXPECT_EQ(1561614240 + TIMEZONE_OFFSET, e5->schedule_time());
  EXPECT_EQ(EventType_DEP, e5->type());
}

char const* reroute_only = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561612800"
  },
  "entity": [{
    "id": "95.TA.59-100-j19-1.91.H",
    "tripUpdate": {
      "trip": {
        "tripId": "95.TA.59-100-j19-1.91.H",
        "startTime": "05:02:00",
        "startDate": "20190627",
        "scheduleRelationship": "SCHEDULED",
        "routeId": "59-100-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "stopId": "8501200:0:4",
        "scheduleRelationship": "SKIPPED"
      }, {
        "stopSequence": 5,
        "stopId": "8501403:0:1",
        "scheduleRelationship": "SKIPPED"
      }, {
        "stopSequence": 7,
        "stopId": "8501506:0:2",
        "scheduleRelationship": "SKIPPED"
      }]
    }
  }]
}
)";

TEST_F(gtfsrt_reroute_test, receive_reroute_only) {
  auto const msgs = parse_json(reroute_only);

  // only a reroute messae is expected
  ASSERT_EQ(1, msgs.size());

  check_reroute_event(msgs[0]);
}

char const* reroute_and_delay = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561612800"
  },
  "entity": [{
    "id": "95.TA.59-100-j19-1.91.H",
    "tripUpdate": {
      "trip": {
        "tripId": "95.TA.59-100-j19-1.91.H",
        "startTime": "05:02:00",
        "startDate": "20190627",
        "scheduleRelationship": "SCHEDULED",
        "routeId": "59-100-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "stopId": "8501200:0:4",
        "scheduleRelationship": "SKIPPED"
      },{
		"stopSequence": 2,
		"departure": {
          "delay": 60
        },
		"stopId": "8501300:0:5",
        "scheduleRelationship": "SCHEDULED"
	  },{
		"stopSequence": 3,
        "arrival": {
          "delay": -60
        },
		"departure": {
          "delay": 120
        }
	  },{
        "stopSequence": 5,
        "stopId": "8501403:0:1",
        "scheduleRelationship": "SKIPPED"
      }, {
        "stopSequence": 7,
        "stopId": "8501506:0:2",
        "scheduleRelationship": "SKIPPED"
      }]
    }
  }]
}
)";

// on stop 3
// also test handling of consecutive sequence numbers without stopid
// and handling of implicit SCHEDULED relationship

TEST_F(gtfsrt_reroute_test, receive_reroute_and_delay) {
  auto const msgs = parse_json(reroute_and_delay);

  // 1. delay and 2. reroute
  ASSERT_EQ(2, msgs.size());

  check_reroute_event(msgs[0]);

  auto const& message = msgs[1];
  EXPECT_EQ(1561612800, message.timestamp_);
  EXPECT_EQ(1561612200 + TIMEZONE_OFFSET, message.earliest_);
  EXPECT_EQ(1561612980 + TIMEZONE_OFFSET, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8501200:0:4", id->station_id()->c_str());
  EXPECT_EQ(1561611720 + TIMEZONE_OFFSET, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(3, events->size());

  auto e0 = events->Get(0)->base();
  EXPECT_STREQ("8501300:0:5", e0->station_id()->c_str());
  EXPECT_EQ(1561612200 + TIMEZONE_OFFSET, e0->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->type());
  EXPECT_EQ(1561612260 + TIMEZONE_OFFSET, events->Get(0)->updated_time());

  auto e1 = events->Get(1)->base();
  EXPECT_STREQ("8501400:0:1", e1->station_id()->c_str());
  EXPECT_EQ(1561612800 + TIMEZONE_OFFSET, e1->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->type());
  EXPECT_EQ(1561612740 + TIMEZONE_OFFSET, events->Get(1)->updated_time());

  auto e2 = events->Get(2)->base();
  EXPECT_STREQ("8501400:0:1", e2->station_id()->c_str());
  EXPECT_EQ(1561612860 + TIMEZONE_OFFSET, e2->schedule_time());
  EXPECT_EQ(EventType_DEP, e2->type());
  EXPECT_EQ(1561612980 + TIMEZONE_OFFSET, events->Get(2)->updated_time());
}

}  // namespace motis::ris::gtfsrt