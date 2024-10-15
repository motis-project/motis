#include "gtest/gtest.h"

#include "motis/core/common/date_time_util.h"
#include "motis/loader/loader.h"
#include "motis/ris/ris_message.h"
#include "motis/test/schedule/gtfs_minimal_poznan.h"

#include "./gtfsrt_test.h"

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_poznan::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_delay_test_poznan : public gtfsrt_test {
public:
  gtfsrt_delay_test_poznan() : gtfsrt_test(dataset_opt) {}
};

constexpr auto const simple_delay_poznan_json = R"(
{
  "header": {
    "gtfsRealtimeVersion": "1.0",
    "incrementality": "FULL_DATASET",
    "timestamp": "1639740232"
  },
  "entity": [
    {
      "id": "5017",
      "tripUpdate": {
        "trip": {
          "tripId": "7_11795^A",
          "scheduleRelationship": "SCHEDULED",
          "routeId": "502"
        },
        "stopTimeUpdate": [
          {
            "stopSequence": 9,
            "arrival": {
              "delay": 60
            },
            "scheduleRelationship": "SCHEDULED"
          }
        ],
        "vehicle": {
          "id": "5017",
          "label": "502/5"
        },
        "timestamp": "1639740222"
      }
    }
  ]
}
)";

TEST_F(gtfsrt_delay_test_poznan, simple_delay_poznan) {
  auto const msgs = parse_json(simple_delay_poznan_json);

  // currently only Is_ Messages and no Forecast expected
  ASSERT_EQ(1, msgs.size());

  auto const& message = msgs[0];
  EXPECT_EQ(1639740232, message.timestamp_);
  EXPECT_EQ(parse_unix_time("2021-12-17 12:26 CET"), message.earliest_);
  EXPECT_EQ(parse_unix_time("2021-12-17 12:27 CET"), message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("3750", id->station_id()->c_str());
  EXPECT_EQ(parse_unix_time("2021-12-17 12:09 CET"), id->schedule_time());
  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(1, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("3955", e0->base()->station_id()->c_str());
  EXPECT_EQ(parse_unix_time("2021-12-17 12:26 CET"),
            e0->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->base()->type());
  EXPECT_EQ(parse_unix_time("2021-12-17 12:27 CET"), e0->updated_time());
}

}  // namespace motis::ris::gtfsrt
