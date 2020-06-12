#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"
#include "../hrd/paths.h"

namespace fbs = flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

auto routing_request_rule_service = R"({
  "destination": {
    "type": "Module",
    "target": "/routing"
  },
  "content_type": "RoutingRequest",
  "content": {
    "start_type": "PretripStart",
    "start": {
      "station": {
        "name": "",
        "id": "0000002"
      },
      "interval": {
        "begin": 1448323200,
        "end": 1448336800
      }
    },
    "destination": {
      "name": "",
      "id": "0000009"
    },
    "additional_edges": [],
    "via": []
  }
})";

auto routing_request_standalone = R"({
  "content_type": "RoutingRequest",
  "content": {
    "start_type": "PretripStart",
    "start": {
      "station": {
        "id": "8000213",
        "name": ""
      },
      "interval":{
        "begin": 1463569200,
        "end": 1463580000
      }
    },
    "destination": {
      "id": "8000297",
      "name": ""
    },
    "search_type": "DefaultForward",
    "additional_edges": [],
    "via": []
  },
  "destination": {
    "type":"Module",
    "target":"/routing"
  }
})";

std::vector<int> trip_train_nrs_at(
    int from, int to, fbs::Vector<fbs::Offset<Trip>> const* trips) {
  std::vector<int> train_nrs;
  for (auto const& t : *trips) {
    if (t->range()->from() < to && t->range()->to() >= from) {
      train_nrs.push_back(t->id()->train_nr());
    }
  }
  std::sort(begin(train_nrs), end(train_nrs));
  return train_nrs;
}

struct loader_graph_builder_rule_service : public motis_instance_test {
  loader_graph_builder_rule_service()
      : motis_instance_test(
            {{(hrd::SCHEDULES / "mss-ts").generic_string()}, "20151124"},
            {"routing"}) {}
};

struct loader_graph_builder_rule_service_standalone
    : public motis_instance_test {
  loader_graph_builder_rule_service_standalone()
      : motis_instance_test(
            {{(hrd::SCHEDULES / "mss-ts-standalone").generic_string()},
             "20160518"},
            {"routing"}) {}
};

TEST_F(loader_graph_builder_rule_service, search) {
  auto res = call(make_msg(routing_request_rule_service));
  auto connections = motis_content(RoutingResponse, res)->connections();

  ASSERT_EQ(1, connections->size());
  for (unsigned i = 0; i < connections->Get(0)->stops()->size() - 2; ++i) {
    EXPECT_FALSE(connections->Get(0)->stops()->Get(i)->exit());
  }

  auto const trips = connections->Get(0)->trips();
  EXPECT_EQ(std::vector<int>({1, 2, 3, 4}), trip_train_nrs_at(0, 5, trips));
  EXPECT_EQ(std::vector<int>({3}), trip_train_nrs_at(0, 1, trips));
  EXPECT_EQ(std::vector<int>({1, 2, 3}), trip_train_nrs_at(1, 4, trips));
}

TEST_F(loader_graph_builder_rule_service_standalone, DISABLED_search) {
  auto res = call(make_msg(routing_request_standalone));
  auto connections = motis_content(RoutingResponse, res)->connections();

  ASSERT_EQ(1, connections->size());
}
