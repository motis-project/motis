#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"
#include "../hrd/paths.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

auto sbb_routing_request = R"({
  "content_type": "RoutingRequest",
  "content": {
    "start_type": "PretripStart",
    "start": {
      "station": {
        "id": "8503000",
        "name": "Zürich HB"
      },
      "interval":{
        "begin": 1585566000,
        "end": 1585584000
      }
    },
    "destination": {
      "id": "8500010",
      "name": "Basel SBB"
    },
    "additional_edges": [],
    "via": []
  },
  "destination": {
    "type":"Module",
    "target":"/routing"
  }
})";

struct loader_sbb : public motis_instance_test {
  loader_sbb()
      : motis_instance_test(
            {{(hrd::SCHEDULES / "sbb").generic_string()}, "20200330"},
            {"routing"}) {}
};

TEST_F(loader_sbb, search) {
  auto res = call(make_msg(sbb_routing_request));
  auto connections = motis_content(RoutingResponse, res)->connections();

  ASSERT_EQ(1, connections->size());
}
