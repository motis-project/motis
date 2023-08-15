#include "gtest/gtest.h"

#include "utl/zip.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/test/motis_instance_test.h"

#include "./resources.h"

namespace fbs = flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

struct loader_graph_builder_gtfs_block_id : public motis_instance_test {
  explicit loader_graph_builder_gtfs_block_id(std::string const& schedule_begin)
      : motis_instance_test(
            loader_options{
                .dataset_ = {(gtfs::SCHEDULES / "block_id").generic_string()},
                .dataset_prefix_ = {"x"},
                .schedule_begin_ = schedule_begin,
                .num_days_ = 2},
            {"routing", "csa", "raptor", "tripbased", "nigiri"},
            {"--tripbased.use_data_file=false", "--nigiri.lookup=false",
             "--nigiri.routing=false",
             fmt::format(
                 "--nigiri.first_day={}-{}-{}", schedule_begin.substr(0, 4),
                 schedule_begin.substr(4, 2), schedule_begin.substr(6, 2)),
             "--nigiri.num_days", "10"}) {}

  msg_ptr routing_query(std::string_view from, std::string_view to,
                        std::string_view start_time, std::string_view target) {
    auto const start_unix_time =
        parse_unix_time(start_time, "%Y-%m-%d %H:%M %Z");
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        routing::CreateRoutingRequest(
            fbb, motis::routing::Start_OntripStationStart,
            routing::CreateOntripStationStart(
                fbb,
                routing::CreateInputStation(
                    fbb,
                    fbb.CreateString(std::string{"x_"} + std::string{from}),
                    fbb.CreateString("")),
                start_unix_time)
                .Union(),
            routing::CreateInputStation(
                fbb, fbb.CreateString(std::string{"x_"} + std::string{to}),
                fbb.CreateString("")),
            routing::SearchType_Default, SearchDir_Forward,
            fbb.CreateVector(std::vector<fbs::Offset<routing::Via>>{}),
            fbb.CreateVector(
                std::vector<fbs::Offset<routing::AdditionalEdgeWrapper>>{}))
            .Union(),
        std::string{target});
    return call(make_msg(fbb));
  }

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  msg_ptr routing_query(std::string_view from, std::string_view to,
                        std::string_view start_time) {
    auto const targets = std::initializer_list<std::string_view>{
        "/routing", "/tripbased", "/raptor_cpu", "/csa", "/nigiri"};
    auto const results = utl::to_vec(targets, [&](auto&& target) {
      return routing_query(from, to, start_time, target);
    });
    auto const reference =
        message_to_journeys(motis_content(RoutingResponse, results.front()));
    for (auto const& [result, target] : utl::zip(results, targets)) {
      std::vector<journey> testee;
      SCOPED_TRACE(fmt::format("{}  ({} --> {})", target, from, to));
      EXPECT_NO_THROW(
          testee = message_to_journeys(motis_content(RoutingResponse, result)));
      EXPECT_EQ(reference, testee);
      if (reference != testee) {
        std::cout << "REF\n";
        for (auto const& x : reference) {
          print_journey(x, std::cout);
        }

        std::cout << "TESTEE\n";
        for (auto const& x : testee) {
          print_journey(x, std::cout);
        }
      }
    }
    return results.front();
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

TEST_F(loader_graph_builder_gtfs_block_id_friday, search_s2_s3) {
  auto res = routing_query("S2", "S3", "2006-07-08 00:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}

TEST_F(loader_graph_builder_gtfs_block_id_friday, search_s2_s7) {
  auto res = routing_query("S2", "S7", "2006-07-08 00:00 Europe/Berlin");
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
