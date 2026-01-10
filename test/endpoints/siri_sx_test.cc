#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>

#include "date/date.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

using namespace motis;
using namespace date;

constexpr auto const kSiriSxGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
TEST,Test Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
STOP1,Stop 1,48.0,9.0,0,,
STOP2,Stop 2,48.1,9.1,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,TEST,R1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,STOP1,1,0,0
T1,10:10:00,10:10:00,STOP2,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20260110,1
)";

constexpr auto const kSiriSxJsonUpdate = R"({
  "responseTimestamp": "2026-01-10T09:55:00Z",
  "estimatedTimetableDelivery": {
    "estimatedJourneyVersionFrame": {
      "recordedAtTime": "2026-01-10T09:55:00Z",
      "estimatedVehicleJourney": {
        "lineRef": "R1",
        "directionRef": "0",
        "framedVehicleJourneyRef": {
          "dataFrameRef": "20260110",
          "datedVehicleJourneyRef": "40-1-24290-78300"
        },
        "estimatedCalls": {
          "estimatedCall": [
            {
              "stopPointRef": {
                "value": "STOP1"
              },
              "order": 1,
              "extraCall": false,
              "cancellation": false,
              "aimedArrivalTime": "2026-01-10T10:00:00+01:00",
              "aimedDepartureTime": "2026-01-10T10:00:00+01:00"
            },
            {
              "stopPointRef": {
                "value": "STOP2"
              },
              "order": 2,
              "extraCall": false,
              "cancellation": false,
              "aimedArrivalTime": "2026-01-10T10:10:00+01:00",
              "aimedDepartureTime": "2026-01-10T10:10:00+01:00"
            }
          ]
        }
      }
    }
  },
  "situationExchangeDelivery": {
    "situations": {
      "ptSituationElement": [
        {
          "creationTime": {
            "value": "2026-01-10T09:00:00Z"
          },
          "situationNumber": {
            "value": "S1"
          },
          "validityPeriod": [
            {
              "startTime": {
                "value": "2026-01-10T09:00:00Z"
              },
              "endTime": {
                "value": "2026-01-10T12:00:00Z"
              }
            }
          ],
          "publicationWindow": {
            "startTime": {
              "value": "2026-01-10T09:00:00Z"
            }
          },
          "severity": {
            "value": "noImpact"
          },
          "affects": {
            "stopPlaces": {
              "affectedStopPlace": [
                {
                  "stopPlaceRef": {
                    "value": "STOP1"
                  }
                }
              ]
            },
            "affectedLines": {
              "affectedLine": [
                {
                  "lineRef": {
                    "value": "4-121-4"
                  }
                },
                {
                  "lineRef": {
                    "value": "4-104-4"
                  }
                }
              ]
            }
          },
          "summary": [
            {
              "value": "Platform change"
            }
          ],
          "description": [
            {
              "value": "Use platform 2"
            }
          ]
        },
        {
          "creationTime": {
            "value": "2026-01-10T09:05:00Z"
          },
          "situationNumber": {
            "value": "S2"
          },
          "validityPeriod": [
            {
              "startTime": {
                "value": "2026-01-10T09:00:00Z"
              },
              "endTime": {
                "value": "2026-01-10T12:00:00Z"
              }
            }
          ],
          "publicationWindow": {
            "startTime": {
              "value": "2026-01-10T09:00:00Z"
            }
          },
          "severity": {
            "value": "minor"
          },
          "affects": {
            "networks": {
              "affectedNetwork": [
                {
                  "affectedLine": [
                    {
                      "lineRef": {
                        "value": "R1"
                      }
                    }
                  ]
                }
              ]
            }
          },
          "summary": [
            {
              "value": "Line disruption"
            }
          ],
          "description": [
            {
              "value": "R1 diverted due to works"
            }
          ]
        },
        {
          "creationTime": {
            "value": "2026-01-10T09:10:00Z"
          },
          "situationNumber": {
            "value": "S3"
          },
          "validityPeriod": [
            {
              "startTime": {
                "value": "2026-01-10T09:00:00Z"
              },
              "endTime": {
                "value": "2026-01-10T12:00:00Z"
              }
            }
          ],
          "publicationWindow": {
            "startTime": {
              "value": "2026-01-10T09:00:00Z"
            }
          },
          "severity": {
            "value": "minor"
          },
          "affects": {
            "vehicleJourneys": {
              "affectedVehicleJourney": [
                {
                  "framedVehicleJourneyRef": {
                    "dataFrameRef": {
                      "value": "20260110"
                    },
                    "datedVehicleJourneyRef": {
                      "value": "40-1-24290-78300"
                    }
                  }
                }
              ]
            }
          },
          "summary": [
            {
              "value": "Vehicle journey issue"
            }
          ],
          "description": [
            {
              "value": "Specific trip impacted"
            }
          ]
        }
      ]
    }
  }
})";

TEST(motis, trip_siri_sx_alerts) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .timetable_ =
          config::timetable{.first_day_ = "2026-01-10",
                            .num_days_ = 1,
                            .datasets_ = {{"test", {.path_ = kSiriSxGtfs}}}},
      .street_routing_ = false};
  auto d = import(c, "test/data", true);
  d.init_rtt(sys_days{2026_y / January / 10});

  auto& rtt = *d.rt_->rtt_;
  auto siri_updater =
      auser(*d.tt_, d.tags_->get_src("test"),
            nigiri::rt::vdv_aus::updater::xml_format::kSiriJson);
  siri_updater.consume_update(kSiriSxJsonUpdate, rtt);

  auto const trip_ep = utl::init_from<ep::trip>(d).value();
  auto const res = trip_ep("?tripId=20260110_10%3A00_test_T1");
  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_.front();
  ASSERT_TRUE(leg.from_.alerts_.has_value());
  ASSERT_FALSE(leg.from_.alerts_->empty());
  EXPECT_EQ("Platform change", leg.from_.alerts_->front().headerText_);
  EXPECT_EQ("Use platform 2", leg.from_.alerts_->front().descriptionText_);
  ASSERT_TRUE(leg.alerts_.has_value());
  auto const has_line_alert = std::any_of(
      begin(*leg.alerts_), end(*leg.alerts_), [](api::Alert const& alert) {
        return alert.headerText_ == "Line disruption" &&
               alert.descriptionText_ == "R1 diverted due to works";
      });
  EXPECT_TRUE(has_line_alert);
  auto const has_vehicle_alert = std::any_of(
      begin(*leg.alerts_), end(*leg.alerts_), [](api::Alert const& alert) {
        return alert.headerText_ == "Vehicle journey issue" &&
               alert.descriptionText_ == "Specific trip impacted";
      });
  EXPECT_TRUE(has_vehicle_alert);
}
