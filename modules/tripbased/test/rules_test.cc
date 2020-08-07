#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;

struct tripbased_rules_test : public motis_instance_test {
  tripbased_rules_test()
      : motis::test::motis_instance_test(
            loader::loader_options{
                {"base/loader/test_resources/hrd_schedules/mss-ts"},
                "20150329",
                3},
            {"tripbased"}, {"--tripbased.use_data_file=false"}) {}
};

TEST_F(tripbased_rules_test, simple_fwd) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString("0000001"),
                                 fbb.CreateString("")),
              unix_time(100))
              .Union(),
          CreateInputStation(fbb, fbb.CreateString("0000010"),
                             fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/tripbased");
  auto const msg = call(make_msg(fbb));
  //  std::cout << msg->to_json() << std::endl;
  auto const res = motis_content(RoutingResponse, msg);
  auto const journeys = message_to_journeys(res);
  ASSERT_EQ(1, journeys.size());
}
