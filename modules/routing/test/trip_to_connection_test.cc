#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"

#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;
using motis::test::schedule::simple_realtime::dataset_opt;

struct routing_trip_to_connection_test : public motis_instance_test {
  routing_trip_to_connection_test()
      : motis::test::motis_instance_test(dataset_opt, {"routing"}) {}
};

TEST_F(routing_trip_to_connection_test, simple) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_TripId,
      CreateTripId(fbb, fbb.CreateString(""), fbb.CreateString("8000096"), 2292,
                   unix_time(1305), fbb.CreateString("8000105"),
                   unix_time(1440), fbb.CreateString("381"))
          .Union(),
      "/trip_to_connection");
  auto const res = call(make_msg(fbb));
  auto const msg = motis_content(Connection, res);
  EXPECT_EQ("8000096", msg->stops()->Get(0)->station()->id()->str());
  EXPECT_EQ(
      "8000105",
      msg->stops()->Get(msg->stops()->Length() - 1)->station()->id()->str());
}
