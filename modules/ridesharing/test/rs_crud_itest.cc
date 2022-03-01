#include "./rs_super_itest.h"

#include "gtest/gtest.h"

#include "motis/ridesharing/routing_result.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace motis::test;
using namespace motis::module;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt_two_weeks;

namespace motis::ridesharing {

struct rs_crud_itest : public motis_instance_test {
  rs_crud_itest()
      : motis::test::motis_instance_test(
            dataset_opt_two_weeks, {"lookup", "ridesharing"},
            {"--ridesharing.database_path=:memory:",
             "--ridesharing.use_parking=false"}) {
    initialize_mocked(*instance_, 12);
  }

  msg_ptr ridesharing_remove(int driver, int time_lift_start) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_RidesharingRemove,
        CreateRidesharingRemove(mc, driver, time_lift_start).Union(),
        "/ridesharing/remove");
    return make_msg(mc);
  }

  msg_ptr ridesharing_unbook(int driver, int time_lift_start, int passenger) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_RidesharingUnbook,
        CreateRidesharingUnbook(mc, driver, time_lift_start, passenger).Union(),
        "/ridesharing/unbook");
    return make_msg(mc);
  }
};

TEST_F(rs_crud_itest, create_book_unbook) {
  publish(make_no_msg("/osrm/initialized"));
  auto res = call(ridesharing_book(123, 200, 321));
  auto content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Found, content->response_type());

  call(ridesharing_create(123, 200));
  call(ridesharing_create(124, 210));

  res = call(ridesharing_book(125, 200, 321));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Found, content->response_type());

  res = call(ridesharing_book(123, 200));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Success, content->response_type());

  res = call(ridesharing_book(123, 200, 321));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Success, content->response_type());

  res = call(ridesharing_unbook(126, 0, 0));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Found, content->response_type());

  res = call(ridesharing_unbook(123, 200, 321));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Success, content->response_type());

  res = call(ridesharing_unbook(123, 200, 321));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Yet_Booked, content->response_type());

  res = call(ridesharing_book(123, 200, 321));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Success, content->response_type());
}

TEST_F(rs_crud_itest, create_remove) {
  publish(make_no_msg("/osrm/initialized"));
  call(ridesharing_create(123, 200));
  call(ridesharing_create(124, 210));

  auto res = call(ridesharing_remove(124, 250));
  auto content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Found, content->response_type());

  res = call(ridesharing_remove(123, 200));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Success, content->response_type());

  res = call(ridesharing_book(123, 200, 1));
  content = motis_content(RidesharingLiftResponse, res);
  ASSERT_EQ(ResponseType_Not_Found, content->response_type());
}

}  // namespace motis::ridesharing
