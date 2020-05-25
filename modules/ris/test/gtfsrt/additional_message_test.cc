#include "gtest/gtest.h"

#include "motis/ris/ris_message.h"
#include "motis/test/schedule/gtfs_minimal_swiss.h"

#include "./gtfsrt_test.h"

#ifdef GetMessage
#undef GetMessage
#endif

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_swiss::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_additional_test : public gtfsrt_test {
public:
  gtfsrt_additional_test() : gtfsrt_test(dataset_opt){};
  static void check_addition_message(ris_message const&);
};

void gtfsrt_additional_test::check_addition_message(
    ris_message const& message) {
  EXPECT_EQ(1561596300, message.timestamp_);
  EXPECT_EQ(1561597920, message.earliest_);
  EXPECT_EQ(1561600800, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_AdditionMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<AdditionMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8501026:0:2", id->station_id()->c_str());
  EXPECT_EQ(1561597920, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(4, events->size());

  auto e0 = events->Get(0)->base();
  EXPECT_STREQ("8501026:0:2", e0->station_id()->c_str());
  EXPECT_EQ(1561597920, e0->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->type());

  auto e1 = events->Get(1)->base();
  EXPECT_STREQ("8501008:0:2", e1->station_id()->c_str());
  EXPECT_EQ(1561599120, e1->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->type());

  auto e2 = events->Get(2)->base();
  EXPECT_STREQ("8501008:0:2", e2->station_id()->c_str());
  EXPECT_EQ(1561599300, e2->schedule_time());
  EXPECT_EQ(EventType_DEP, e2->type());

  auto e3 = events->Get(3)->base();
  EXPECT_STREQ("8503000:0:14", e3->station_id()->c_str());
  EXPECT_EQ(1561600800, e3->schedule_time());
  EXPECT_EQ(EventType_ARR, e3->type());
}

const char* additional_only = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "j19-odp-06004-H-A.65528",
    "tripUpdate": {
      "trip": {
        "tripId": "j19-odp-06004-H-A.65528",
        "startTime": "01:12:00",
        "startDate": "20190627",
        "scheduleRelationship": "ADDED",
        "routeId": "9-1-A-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "departure": {
          "time": "1561597920"
        },
        "stopId": "8501026:0:2",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 2,
        "arrival": {
          "time": "1561599120"
        },
        "departure": {
          "time": "1561599300"
        },
        "stopId": "8501008:0:2",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 3,
        "arrival": {
          "time": "1561600800"
        },
        "stopId": "8503000:0:14",
        "scheduleRelationship": "SCHEDULED"
      }]
    }
  }]
}
)";

TEST_F(gtfsrt_additional_test, receive_additional_only) {
  auto const msgs = parse_json(additional_only);

  ASSERT_EQ(1, msgs.size());
  check_addition_message(msgs[0]);
}

const char* additional_and_delay = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "j19-odp-06004-H-A.65528",
    "tripUpdate": {
      "trip": {
        "tripId": "j19-odp-06004-H-A.65528",
        "startTime": "01:12:00",
        "startDate": "20190627",
        "scheduleRelationship": "ADDED",
        "routeId": "9-1-A-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "departure": {
          "time": "1561597920"
        },
        "stopId": "8501026:0:2",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 2,
        "arrival": {
		  "delay": 0,
          "time": "1561599120"
        },
        "departure": {
		  "delay": -120,
          "time": "1561599300"
        },
        "stopId": "8501008:0:2",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 3,
        "arrival": {
		  "delay": 0,
          "time": "1561600800"
        },
        "stopId": "8503000:0:14",
        "scheduleRelationship": "SCHEDULED"
      }]
    }
  }]
}
)";

TEST_F(gtfsrt_additional_test, receive_additional_and_delay) {
  auto const msgs = parse_json(additional_and_delay);
  ASSERT_EQ(2, msgs.size());
  check_addition_message(msgs[0]);

  auto const& message = msgs[1];
  EXPECT_EQ(1561596300, message.timestamp_);
  EXPECT_EQ(1561599120, message.earliest_);
  EXPECT_EQ(1561600800, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8501026:0:2", id->station_id()->c_str());
  EXPECT_EQ(1561597920, id->schedule_time());
  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(3, events->size());

  auto e0 = events->Get(0)->base();
  EXPECT_STREQ("8501008:0:2", e0->station_id()->c_str());
  EXPECT_EQ(1561599120, e0->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->type());
  EXPECT_EQ(1561599120, events->Get(0)->updated_time());

  auto e1 = events->Get(1)->base();
  EXPECT_STREQ("8501008:0:2", e1->station_id()->c_str());
  EXPECT_EQ(1561599300, e1->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->type());
  EXPECT_EQ(1561599180, events->Get(1)->updated_time());

  auto e2 = events->Get(2)->base();
  EXPECT_STREQ("8503000:0:14", e2->station_id()->c_str());
  EXPECT_EQ(1561600800, e2->schedule_time());
  EXPECT_EQ(EventType_ARR, e2->type());
  EXPECT_EQ(1561600800, events->Get(2)->updated_time());
}

const char* additional_delay_reroute = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "timestamp": "1561596300"
  },
  "entity": [{
    "id": "j19-odp-06004-H-A.65528",
    "tripUpdate": {
      "trip": {
        "tripId": "j19-odp-06004-H-A.65528",
        "startTime": "01:12:00",
        "startDate": "20190627",
        "scheduleRelationship": "ADDED",
        "routeId": "9-1-A-j19-1"
      },
      "stopTimeUpdate": [{
        "stopSequence": 1,
        "stopId": "8501026:0:2",
        "scheduleRelationship": "SKIPPED"
      }, {
        "stopSequence": 2,
        "arrival": {
		  "delay": 0,
          "time": "1561599120"
        },
        "departure": {
		  "delay": -120,
          "time": "1561599300"
        },
        "stopId": "8501008:0:2",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 3,
        "stopId": "8503000:0:14",
        "scheduleRelationship": "SKIPPED"
      }, {
        "stopSequence": 4,
        "arrival": {
		  "delay": 0,
          "time": "1561600800"
        },
        "departure": {
          "time": "1561597920"
        },
        "stopId": "8503000:0:14",
        "scheduleRelationship": "SCHEDULED"
      }, {
        "stopSequence": 5,
        "stopId": "8503000:0:14",
        "scheduleRelationship": "SKIPPED"
      }]
    }
  }]
}
)";

TEST_F(gtfsrt_additional_test, receive_addition_delay_reroute) {
  auto const msgs = parse_json(additional_delay_reroute);

  ASSERT_EQ(2, msgs.size());

  auto const& addition_msg = msgs[0];
  EXPECT_EQ(1561596300, addition_msg.timestamp_);
  EXPECT_EQ(1561599300, addition_msg.earliest_);
  EXPECT_EQ(1561600800, addition_msg.latest_);

  // addition Message
  auto outer_addition = GetMessage(addition_msg.data());
  ASSERT_EQ(MessageUnion_AdditionMessage, outer_addition->content_type());
  auto inner_addition =
      reinterpret_cast<AdditionMessage const*>(outer_addition->content());

  auto id = inner_addition->trip_id();
  EXPECT_STREQ("8501008:0:2", id->station_id()->c_str());
  EXPECT_EQ(1561599300, id->schedule_time());

  auto addition_events = inner_addition->events();
  ASSERT_EQ(2, addition_events->size());

  auto e0 = addition_events->Get(0)->base();
  EXPECT_STREQ("8501008:0:2", e0->station_id()->c_str());
  EXPECT_EQ(1561599300, e0->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->type());

  auto e1 = addition_events->Get(1)->base();
  EXPECT_STREQ("8503000:0:14", e1->station_id()->c_str());
  EXPECT_EQ(1561600800, e1->schedule_time());
  EXPECT_EQ(EventType_ARR, e1->type());

  auto const& delay_msg = msgs[1];
  EXPECT_EQ(1561596300, delay_msg.timestamp_);
  EXPECT_EQ(1561599180, delay_msg.earliest_);
  EXPECT_EQ(1561600800, delay_msg.latest_);

  // Delay Message
  auto outer_delay = GetMessage(delay_msg.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_delay->content_type());
  auto inner_delay =
      reinterpret_cast<DelayMessage const*>(outer_delay->content());

  id = inner_addition->trip_id();
  EXPECT_STREQ("8501008:0:2", id->station_id()->c_str());
  EXPECT_EQ(1561599300, id->schedule_time());

  auto delay_events = inner_delay->events();
  ASSERT_EQ(2, delay_events->size());

  e0 = delay_events->Get(0)->base();
  EXPECT_STREQ("8501008:0:2", e0->station_id()->c_str());
  EXPECT_EQ(1561599300, e0->schedule_time());
  EXPECT_EQ(1561599180, delay_events->Get(0)->updated_time());
  EXPECT_EQ(EventType_DEP, e0->type());

  e1 = delay_events->Get(1)->base();
  EXPECT_STREQ("8503000:0:14", e1->station_id()->c_str());
  EXPECT_EQ(1561600800, e1->schedule_time());
  EXPECT_EQ(1561600800, delay_events->Get(1)->updated_time());
  EXPECT_EQ(EventType_ARR, e1->type());
}

}  // namespace motis::ris::gtfsrt