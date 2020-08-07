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

struct loader_graph_builder_gtfs_block_id : public motis_instance_test {
  explicit loader_graph_builder_gtfs_block_id(std::string schedule_begin)
      : motis_instance_test({{(gtfs::SCHEDULES / "block_id").generic_string()},
                             std::move(schedule_begin),
                             1},
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
            fbb.CreateVector(std::vector<fbs::Offset<routing::Via>>{}),
            fbb.CreateVector(
                std::vector<fbs::Offset<routing::AdditionalEdgeWrapper>>{}))
            .Union(),
        "/routing");
    return call(make_msg(fbb));
  }

  static void expect_no_transfers(Connection const* con) {
    for (unsigned i = 0; i < con->stops()->size() - 2; ++i) {
      EXPECT_FALSE(con->stops()->Get(i)->exit());
    }
  }
};

struct loader_graph_builder_gtfs_block_id_once
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_once()
      : loader_graph_builder_gtfs_block_id{"20060702"} {}
};

struct loader_graph_builder_gtfs_block_id_monday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_monday()
      : loader_graph_builder_gtfs_block_id{"20060703"} {}
};

struct loader_graph_builder_gtfs_block_id_tuesday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_tuesday()
      : loader_graph_builder_gtfs_block_id{"20060704"} {}
};

struct loader_graph_builder_gtfs_block_id_wednesday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_wednesday()
      : loader_graph_builder_gtfs_block_id{"20060705"} {}
};

struct loader_graph_builder_gtfs_block_id_thursday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_thursday()
      : loader_graph_builder_gtfs_block_id{"20060706"} {}
};

struct loader_graph_builder_gtfs_block_id_friday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_friday()
      : loader_graph_builder_gtfs_block_id{"20060707"} {}
};

struct loader_graph_builder_gtfs_block_id_saturday
    : public loader_graph_builder_gtfs_block_id {
  loader_graph_builder_gtfs_block_id_saturday()
      : loader_graph_builder_gtfs_block_id{"20060708"} {}
};

TEST_F(loader_graph_builder_gtfs_block_id_once, search_s1_s8) {
  auto res = routing_query("S1", "S8", "2006-07-02 23:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_once, search_s2_s1) {
  auto res = routing_query("S2", "S1", "2006-07-02 23:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(0, conns->size());
}

TEST_F(loader_graph_builder_gtfs_block_id_saturday, search_s2_s3) {
  auto res = routing_query("S2", "S3", "2006-07-09 00:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_saturday, search_s2_s7) {
  auto res = routing_query("S2", "S7", "2006-07-09 00:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_wednesday, search_s1_s4) {
  auto res = routing_query("S1", "S4", "2006-07-05 23:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_thursday, search_s1_s5) {
  auto res = routing_query("S1", "S5", "2006-07-06 23:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_friday, search_s1_s7) {
  auto res = routing_query("S1", "S7", "2006-07-07 23:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}