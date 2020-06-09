#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"

#include "./schedule.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::railviz;

struct railviz_get_trip_guesses_test : public motis_instance_test {
  railviz_get_trip_guesses_test()
      : motis::test::motis_instance_test(dataset_opt, {"railviz"}) {}

  static msg_ptr get_trip_guesses(int const train_num, time_t const t) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RailVizTripGuessRequest,
        CreateRailVizTripGuessRequest(fbb, train_num, t, train_num).Union(),
        "/railviz/get_trip_guesses");
    return make_msg(fbb);
  }
};

TEST_F(railviz_get_trip_guesses_test, simple_result_ok) {
  auto const res_msg = call(get_trip_guesses(1, unix_time(1200)));
  auto const res = motis_content(RailVizTripGuessResponse, res_msg);
  EXPECT_EQ(1, res->trips()->size());
  EXPECT_EQ(1, res->trips()->Get(0)->trip_info()->id()->train_nr());
}
