#include "gtest/gtest.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::routing;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace {
loader::loader_options add_tag(loader::loader_options opt) {
  opt.dataset_prefix_.emplace_back("x");
  return opt;
}
}  // namespace

struct routing_itest : public motis_instance_test {
  routing_itest()
      : motis::test::motis_instance_test(
            add_tag(dataset_opt),
            {"routing", "csa", "raptor", "tripbased", "nigiri"},
            {"--tripbased.use_data_file=false", "--nigiri.lookup=false",
             "--nigiri.routing=false", "--nigiri.first_day=2015-11-24"}) {}

  msg_ptr make_routing_request(std::string const& target) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            fbb, motis::routing::Start_OntripStationStart,
            CreateOntripStationStart(
                fbb,
                CreateInputStation(fbb, fbb.CreateString("x_8000096"),
                                   fbb.CreateString("")),
                unix_time(1300))
                .Union(),
            CreateInputStation(fbb, fbb.CreateString("x_8000080"),
                               fbb.CreateString("")),
            SearchType_Default, SearchDir_Forward,
            fbb.CreateVector(std::vector<Offset<Via>>()),
            fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
            .Union(),
        target);
    return make_msg(fbb);
  }
};

TEST_F(routing_itest, all_routings_deliver_equal_journey) {
  auto const reference = message_to_journeys(
      motis_content(RoutingResponse, call(make_routing_request("/routing"))));
  EXPECT_EQ(reference.size(), 1U);
  for (auto const& target :
       {"/routing", "/tripbased", "/raptor_cpu", "/csa", "/nigiri"}) {
    SCOPED_TRACE(target);

    std::vector<journey> testee;
    EXPECT_NO_THROW(testee = message_to_journeys(motis_content(
                        RoutingResponse, call(make_routing_request(target)))));
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
    EXPECT_EQ(reference, testee);
  }
}
