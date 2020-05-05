#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"

#include "motis/railviz/geo.h"

#include "./schedule.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::railviz;

struct railviz_get_trains_test : public motis_instance_test {
  railviz_get_trains_test()
      : motis::test::motis_instance_test(dataset_opt, {"railviz"}) {}

  static msg_ptr get_trains(geo::coord const c1, geo::coord const c2,  //
                            time_t const min, time_t const max) {
    Position const p1(c1.lat_, c1.lng_);
    Position const p2(c2.lat_, c2.lng_);
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_RailVizTrainsRequest,
        CreateRailVizTrainsRequest(fbb, 16, 18, &p1, &p2, min, max, 100)
            .Union(),
        "/railviz/get_trains");
    return make_msg(fbb);
  }
};

TEST_F(railviz_get_trains_test, simple_result_ok) {
  using namespace std::string_literals;

  auto const min = unix_time(1215);
  auto const max = unix_time(1220);
  auto const res_msg = call(
      get_trains({50.322747, 9.0133358}, {50.322747, 9.0133358}, min, max));

  auto const res = motis_content(RailVizTrainsResponse, res_msg);

  for (auto const& t : *res->trains()) {
    auto const segment = res->routes()
                             ->Get(t->route_index())
                             ->segments()
                             ->Get(t->segment_index());

    auto const stations = std::make_pair(segment->from_station_id()->str(),
                                         segment->to_station_id()->str());
    EXPECT_TRUE(stations == std::make_pair("5386096"s, "7347220"s) ||
                stations == std::make_pair("7347220"s, "7190994"s));

    EXPECT_GE(min, t->d_time());
    EXPECT_LE(max, t->a_time());
    EXPECT_LE(t->d_time(), t->a_time());
  }

  EXPECT_EQ(4U, res->trains()->size());
}
