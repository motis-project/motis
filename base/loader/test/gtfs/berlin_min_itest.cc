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

struct loader_graph_builder_gtfs_berlin_min : public motis_instance_test {
  explicit loader_graph_builder_gtfs_berlin_min()
      : motis_instance_test(
            loader_options{
                .dataset_ = {(gtfs::SCHEDULES / "berlin_min").generic_string()},
                .dataset_prefix_ = {"x"},
                .schedule_begin_ = std::string{schedule_begin},
                .num_days_ = 2},
            {"routing", "nigiri"},
            {"--nigiri.num_days=2", "--nigiri.routing=false",
             "--nigiri.lookup=false",
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

TEST_F(loader_graph_builder_gtfs_berlin_min, search) {
  auto const& s = sched();
  for (auto const& t : s.trips_) {
    print_trip(std::cout, s, t.second, false);
  }

  std::cout << "digraph {\n";
  for (auto const& sn : s.station_nodes_) {
    if (sn->child_nodes_.empty()) {
      continue;
    }
    std::cout << "  subgraph cluster_" << sn->id_ << " {\n";
    std::cout << "    label=\"" << s.stations_[sn->id_]->name_ << "\"\n    ";
    sn->for_each_route_node(
        [&](node const* n) { std::cout << n->id_ << "; "; });
    std::cout << "\n  }\n";
  }
  for (auto const& sn : s.station_nodes_) {
    sn->for_each_route_node([&](node const* n) {
      for (auto const& e : n->edges_) {
        if (e.empty() && e.type() != edge::THROUGH_EDGE) {
          continue;
        }
        std::cout << "  " << e.from_->id_ << " -> " << e.to_->id_;
        if (e.type() == edge::THROUGH_EDGE) {
          std::cout << " [color=red,penwidth=3.0];";
        } else {
          std::cout << ";";
        }
        std::cout << "\n";
      }
    });
  }
  std::cout << "}\n";

  auto res = routing_query("de:11000:900160548::1", "de:11000:900175013::1",
                           "2023-04-16 20:00 Europe/Berlin");
  auto conns = motis_content(RoutingResponse, res)->connections();
  ASSERT_EQ(0, conns->size());
}