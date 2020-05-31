#include "gtest/gtest.h"

#include "motis/core/common/date_time_util.h"
#include "motis/test/motis_instance_test.h"

#include "./resources.h"

namespace fbs = flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

auto routing_request_gtfs_block_id = R"({
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

struct loader_graph_builder_gtfs_block_id : public motis_instance_test {
  loader_graph_builder_gtfs_block_id()
      : motis_instance_test(
            {(gtfs::SCHEDULES / "block_id").generic_string(), "20060701", 31},
            {"routing"}) {}

  msg_ptr routing_query(std::string_view const& from,
                        std::string_view const& to,
                        std::string_view start_time) {
    auto const start_unix_time =
        parse_unix_time(start_time, "%Y-%m-%d %H:%M %Z");
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        routing::CreateRoutingRequest(
            fbb, motis::routing::Start_OntripStationStart,
            routing::CreateOntripStationStart(
                fbb,
                routing::CreateInputStation(fbb, fbb.CreateString(from),
                                            fbb.CreateString("")),
                start_unix_time)
                .Union(),
            routing::CreateInputStation(fbb, fbb.CreateString(to),
                                        fbb.CreateString("")),
            routing::SearchType_Default, routing::SearchDir_Forward,
            fbb.CreateVector(std::vector<flatbuffers::Offset<routing::Via>>{}),
            fbb.CreateVector(
                std::vector<
                    flatbuffers::Offset<routing::AdditionalEdgeWrapper>>{}))
            .Union(),
        "/routing");
    return call(make_msg(fbb));
  }
};

TEST_F(loader_graph_builder_gtfs_block_id, search) {
  auto res = routing_query("S1", "S2", "2006-07-02 01:00 Europe/Berlin");
  auto connections = motis_content(RoutingResponse, res)->connections();

  ASSERT_EQ(1, connections->size());
  std::cout << res->to_json() << "\n";
  for (unsigned i = 0; i < connections->Get(0)->stops()->size() - 2; ++i) {
    EXPECT_FALSE(connections->Get(0)->stops()->Get(i)->exit());
  }
}