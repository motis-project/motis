#include "gtest/gtest.h"

#include "motis/loader/loader.h"
#include "motis/ris/ris_message.h"
#include "motis/test/schedule/gtfs_minimal_poznan.h"

#include "./gtfsrt_test.h"

using namespace motis;
using namespace motis::test;
using motis::test::schedule::gtfs_minimal_poznan::dataset_opt;

namespace motis::ris::gtfsrt {

class gtfsrt_delay_test : public gtfsrt_test {
public:
  gtfsrt_delay_test() : gtfsrt_test(dataset_opt) {}
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
      "vehicle": {
        "trip": {
          "tripId": "7_11795^A",
          "scheduleRelationship": "SCHEDULED",
          "routeId": "502"
        },
        "position": {
          "latitude": 52.2534599,
          "longitude": 17.0892696,
          "speed": 4.44
        },
        "currentStopSequence": 9,
        "timestamp": "1639740222",
        "vehicle": {
          "id": "5017",
          "label": "502/5"
        }
      }
    },
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
              "delay": 20
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

constexpr auto const TIMEZONE_OFFSET = -7200;

TEST_F(gtfsrt_delay_test, simple_delay_poznan) {
  auto const msgs = parse_json(simple_delay_poznan_json);

  // currently only Is_ Messages and no Forecast expected
  ASSERT_EQ(1, msgs.size());
}

}  // namespace motis::ris::gtfsrt