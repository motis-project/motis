#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"
#include "../hrd/paths.h"

#include "motis/core/access/station_access.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

struct multischedule_test : public motis_instance_test {
  multischedule_test()
      : motis_instance_test({{(hrd::SCHEDULES / "single-ice").generic_string(),
                              (hrd::SCHEDULES / "single-bus").generic_string()},
                             "20151004"},
                            {"routing"}) {}
};

TEST_F(multischedule_test, stations) {
  {  // some stations from "single-ice"
    EXPECT_NE(nullptr, find_station(sched(), "8000261"));  // München Hbf
    EXPECT_NE(nullptr, find_station(sched(), "8011102"));  // Gesundbrunnen
    EXPECT_NE(nullptr, find_station(sched(), "8000122"));  // Treuchtlingen
  }

  {  // some stations from "single-bus"
    EXPECT_NE(nullptr, find_station(sched(), "0460350"));  // Gunzenhausen
    EXPECT_NE(nullptr, find_station(sched(), "0683474"));  // Treuchtlingen
  }
}

TEST_F(multischedule_test, search) {
  {  // search in "single-ice" München -> Berlin
    auto res = call(make_msg(R"({
      "content_type": "RoutingRequest",
      "content": {
        "start_type": "OntripStationStart",
        "start": {
          "station": { "id": "8000261", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "8098160", "name": "" },
        "additional_edges": [],  "via": []
      },
      "destination": {
        "type":"Module",
        "target":"/routing"
      }
    })"));
    EXPECT_EQ(1, motis_content(RoutingResponse, res)->connections()->size());
  }
  {  // search in "single-ice" Gunzenhausen -> Treuchtlingen
    auto res = call(make_msg(R"({
      "content_type": "RoutingRequest",
      "content": {
        "start_type": "OntripStationStart",
        "start": {
          "station": { "id": "0460350", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "0683474", "name": "" },
        "additional_edges": [],  "via": []
      },
      "destination": {
        "type":"Module",
        "target":"/routing"
      }
    })"));
    EXPECT_EQ(1, motis_content(RoutingResponse, res)->connections()->size());
  }
  {  // search in both Gunzenhausen -> Berlin
    auto res = call(make_msg(R"({
      "content_type": "RoutingRequest",
      "content": {
        "start_type": "OntripStationStart",
        "start": {
          "station": { "id": "0460350", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "8098160", "name": "" },
        "additional_edges": [],  "via": []
      },
      "destination": {
        "type":"Module",
        "target":"/routing"
      }
    })"));
    EXPECT_EQ(1, motis_content(RoutingResponse, res)->connections()->size());
  }
}
