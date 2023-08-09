#include "gtest/gtest.h"

#include "utl/zip.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/test/motis_instance_test.h"

#include "motis/core/journey/print_trip.h"
#include "./resources.h"

namespace fbs = flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;
using motis::routing::RoutingResponse;

constexpr auto const schedule_begin = std::string_view{"20230415"};

struct loader_graph_builder_gtfs_berlin_block_id_day_change
    : public motis_instance_test {
  explicit loader_graph_builder_gtfs_berlin_block_id_day_change()
      : motis_instance_test(
            loader_options{
                .dataset_ = {(gtfs::SCHEDULES / "berlin_block_id_day_change")
                                 .generic_string()},
                .dataset_prefix_ = {"x"},
                .schedule_begin_ = std::string{schedule_begin},
                .num_days_ = 2},
            {"routing", "nigiri"},
            {"--nigiri.num_days=2", "--nigiri.lookup=false",
             "--nigiri.routing=false",
             fmt::format(
                 "--nigiri.first_day={}-{}-{}", schedule_begin.substr(0, 4),
                 schedule_begin.substr(4, 2), schedule_begin.substr(6, 2))}) {}

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
    auto const targets =
        std::initializer_list<std::string_view>{"/routing", "/nigiri"};
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
          print_journey(x, std::cout, false);
        }

        std::cout << "TESTEE\n";
        for (auto const& x : testee) {
          print_journey(x, std::cout, false);
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

TEST_F(loader_graph_builder_gtfs_berlin_block_id_day_change, search) {
  auto res = routing_query("de:11000:900175002::2", "de:11000:900195524::1",
                           "2023-04-16 00:14 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(1, conns->size());
  expect_no_transfers(conns->Get(0));
}