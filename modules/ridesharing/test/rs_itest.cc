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
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::ridesharing {

struct rs_itest : public rs_super_itest {
  rs_itest() : rs_super_itest(10000) {}
};

TEST_F(rs_itest, edges_book) {
  message_creator mc;
  publish(make_no_msg("/osrm/initialized"));
  call(ridesharing_create(123, 1500, 7.2));
  auto res = call(ridesharing_edges());
  auto content = motis_content(RidesharingResponse, res);

  // close-station-radius = 50000
  ASSERT_EQ(6, content->arrs()->size());
  ASSERT_EQ(3, content->deps()->size());
  ASSERT_EQ(1, content->direct_connections()->size());

  auto const* test_edge = content->deps()->Get(0);

  ASSERT_EQ("START", test_edge->from_station_id()->str());
  ASSERT_EQ("8000084", test_edge->to_station_id()->str());
  ASSERT_EQ(3560, test_edge->rs_duration());
  ASSERT_EQ(518, test_edge->rs_price());
  ASSERT_EQ(0, test_edge->ppr_accessibility());
  ASSERT_EQ(3500, test_edge->rs_t());

  Position pick_up{50.8, 6.5};
  Position drop_off{50.8, 6.6};

  mc.create_and_finish(MsgContent_RidesharingBook,
                       CreateRidesharingBook(mc, 123, 1500, 3456, 1, 100000,
                                             &pick_up, 0, &drop_off, 0, 250)
                           .Union(),
                       "/ridesharing/book");
  call(make_msg(mc));
  res = call(ridesharing_edges());
  content = motis_content(RidesharingResponse, res);

  ASSERT_EQ(6, content->arrs()->size());
  ASSERT_EQ(3, content->deps()->size());
  ASSERT_EQ(1, content->direct_connections()->size());

  auto const* test_edge2 = content->deps()->Get(2);

  ASSERT_EQ("START", test_edge2->from_station_id()->str());
  ASSERT_EQ("8005556", test_edge2->to_station_id()->str());
  ASSERT_EQ(12145, test_edge2->rs_duration());
  ASSERT_EQ(1707, test_edge2->rs_price());
  ASSERT_EQ(0, test_edge2->ppr_accessibility());
  ASSERT_EQ(3500, test_edge2->rs_t());

  pick_up = {50.79, 6.7};
  drop_off = {50.79, 6.9};

  mc.create_and_finish(MsgContent_RidesharingBook,
                       CreateRidesharingBook(mc, 123, 1500, 7890, 1, 97500,
                                             &pick_up, 2, &drop_off, 2, 250)
                           .Union(),
                       "/ridesharing/book");
  call(make_msg(mc));
  res = call(ridesharing_edges());
  content = motis_content(RidesharingResponse, res);

  ASSERT_EQ(5, content->arrs()->size());
  ASSERT_EQ(3, content->deps()->size());
  ASSERT_EQ(1, content->direct_connections()->size());
}

TEST_F(rs_itest, edges_multiple_lift) {
  message_creator mc;

  // close-station-radius = 50000
  publish(make_no_msg("/osrm/initialized"));
  call(ridesharing_create(123, 1500, 7.2));
  call(ridesharing_create(123, 1000, 7.5));

  auto res = call(ridesharing_edges(50.76));
  auto content = motis_content(RidesharingResponse, res);
  ASSERT_EQ(7, content->arrs()->size());
  ASSERT_EQ(2, content->direct_connections()->size());
}

}  // namespace motis::ridesharing
