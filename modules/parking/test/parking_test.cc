#include "gtest/gtest.h"

#include <iostream>
#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "geo/latlng.h"

#include "motis/module/message.h"

#include "motis/test/motis_instance_test.h"

#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace geo;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::parking;
using namespace motis::test;
using namespace motis::osrm;
using namespace motis::ppr;

namespace motis::parking {

struct parking_test : public motis_instance_test {
  parking_test()
      : motis::test::motis_instance_test(
            motis::test::schedule::simple_realtime::dataset_opt, {"parking"},
            {"--parking.parking=modules/"
             "parking/test_resources/"
             "parking.txt",
             "--parking.db_max_size=1048576", "--parking.db=-"}) {}
};

TEST_F(parking_test, first) {
  message_creator fbb;
  fbb.create_and_finish(MsgContent_ParkingLookupRequest,
                        CreateParkingLookupRequest(fbb, 1).Union(),
                        "/parking/lookup");
  auto const res = call(make_msg(fbb));
  auto const msg = motis_content(ParkingLookupResponse, res);
  EXPECT_EQ(1, msg->parking()->id());
  EXPECT_EQ(49.87111, msg->parking()->pos()->lat());
  EXPECT_EQ(8.63464, msg->parking()->pos()->lng());
}

TEST_F(parking_test, invalid1) {
  message_creator fbb;
  fbb.create_and_finish(MsgContent_ParkingLookupRequest,
                        CreateParkingLookupRequest(fbb, 0).Union(),
                        "/parking/lookup");
  ASSERT_THROW(call(make_msg(fbb)), std::system_error);
}

TEST_F(parking_test, invalid2) {
  message_creator fbb;
  fbb.create_and_finish(MsgContent_ParkingLookupRequest,
                        CreateParkingLookupRequest(fbb, -1).Union(),
                        "/parking/lookup");
  ASSERT_THROW(call(make_msg(fbb)), std::system_error);
}

TEST_F(parking_test, invalid3) {
  message_creator fbb;
  fbb.create_and_finish(MsgContent_ParkingLookupRequest,
                        CreateParkingLookupRequest(fbb, 10000).Union(),
                        "/parking/lookup");
  ASSERT_THROW(call(make_msg(fbb)), std::system_error);
}

}  // namespace motis::parking
