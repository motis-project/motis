#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"

#include "./schedule.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::railviz;

struct railviz_get_station_test : public motis_instance_test {
  railviz_get_station_test()
      : motis::test::motis_instance_test(dataset_opt, {"railviz"}) {}

  static msg_ptr get_station(std::string const& station_id, time_t const t,
                             unsigned event_count) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RailVizStationRequest,
        CreateRailVizStationRequest(fbb, fbb.CreateString(station_id), t,
                                    event_count, Direction_BOTH, true)
            .Union(),
        "/railviz/get_station");
    return make_msg(fbb);
  }
};

TEST_F(railviz_get_station_test, simple_result_ok) {
  auto const res_msg = call(get_station("5386096", unix_time(1500), 3));
  auto const res = motis_content(RailVizStationResponse, res_msg);
  EXPECT_EQ(res->station()->name()->str(), "Darmstadt hbf");
  EXPECT_EQ(3, res->events()->size());
}
