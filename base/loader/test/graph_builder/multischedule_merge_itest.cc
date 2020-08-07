#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"
#include "../hrd/paths.h"

#include "motis/core/access/station_access.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

struct multischedule_merge_test : public motis_instance_test {
  multischedule_merge_test()
      : motis_instance_test(
            loader_options{
                .dataset_ = {(hrd::SCHEDULES / "single-ice").generic_string(),
                             (hrd::SCHEDULES / "single-bus").generic_string()},
                .schedule_begin_ = "20151004",
                .dataset_prefix_ = {{"ice"}, {"bus"}}},
            {"routing"}) {}
};

TEST_F(multischedule_merge_test, stations) {
  {  // some stations from "single-ice"
    EXPECT_NE(nullptr, find_station(sched(), "ice_8000261"));  // München Hbf
    EXPECT_NE(nullptr, find_station(sched(), "ice_8011102"));  // Gesundbrunnen
    EXPECT_NE(nullptr, find_station(sched(), "ice_8000122"));  // Treuchtlingen
  }

  {  // some stations from "single-bus"
    EXPECT_NE(nullptr, find_station(sched(), "bus_0460350"));  // Gunzenhausen
    EXPECT_NE(nullptr, find_station(sched(), "bus_0683474"));  // Treuchtlingen
  }
}

TEST_F(multischedule_merge_test, search) {
  {  // search in "single-ice" München -> Berlin
    auto res = call(make_msg(R"({
      "content_type": "RoutingRequest",
      "content": {
        "start_type": "OntripStationStart",
        "start": {
          "station": { "id": "ice_8000261", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "ice_8098160", "name": "" },
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
          "station": { "id": "bus_0460350", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "bus_0683474", "name": "" },
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
          "station": { "id": "bus_0460350", "name": "" },
          "departure_time": 1443970800
        },
        "destination": { "id": "ice_8098160", "name": "" },
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
