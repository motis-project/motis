#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"

#include "motis/module/message.h"

#include "motis/ridesharing/routing_result.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

#include <string>

#include "./rs_super_itest.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"
#include "gtest/gtest.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::parking;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::ppr;
using namespace motis::intermodal;
using motis::test::schedule::simple_realtime::dataset_opt_two_weeks;

namespace motis::ridesharing {

struct rs_parking_itest : public motis_instance_test {
  rs_parking_itest()
      : motis::test::motis_instance_test(
            dataset_opt_two_weeks, {"lookup", "ridesharing", "parking"},
            {"--ridesharing.database_path=:memory:",
             "--ridesharing.use_parking=true",
             "--parking.parking=modules/parking/test_resources/"
             "parking_ridesharing.txt"}) {
    initialize_mocked(*instance_, 100000);
  }
};

TEST_F(rs_parking_itest, DISABLED_edges_multiple_lift) {
  // TODO(root) not working because initialization of ppr requires OSM file
  // and parking requires pppr
  message_creator mc;

  // close-station-radius = 50000
  publish(make_no_msg("/osrm/initialized"));
  call(ridesharing_create(123, 1500, 7.2));

  auto res = call(ridesharing_edges(50.78));
  auto content = motis_content(RidesharingResponse, res);
  ASSERT_EQ(6, content->arrs()->size());
  ASSERT_EQ(2, content->deps()->size());
  ASSERT_EQ(1, content->direct_connections()->size());
  auto const* test_edge = content->arrs()->Get(0);
  ASSERT_EQ("END", test_edge->to_station_id()->str());
  auto const* test_edge2 = content->deps()->Get(0);

  ASSERT_EQ("START", test_edge2->from_station_id()->str());
  ASSERT_EQ(2, test_edge2->ppr_accessibility());
  ASSERT_EQ(3, test_edge2->ppr_duration());
}

}  // namespace motis::ridesharing
