#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"

#include "motis/core/access/station_access.h"

#include "motis/railviz/train_retriever.h"

#include "./schedule.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::railviz;

struct railviz_train_retriever_test : public motis_instance_test {
  railviz_train_retriever_test()
      : motis::test::motis_instance_test(dataset_opt) {}
};

TEST_F(railviz_train_retriever_test, simple_result_ok) {
  auto const& s = sched();

  auto const t_min = unix_to_motistime(s, unix_time(1215));
  auto const t_max = unix_to_motistime(s, unix_time(1220));

  geo::box box;
  for (auto const& s : s.stations_) {
    if (s->lat() != 0. && s->lng() != 0.) {
      box.extend({s->lat(), s->lng()});
    }
  }
  float const expected_distance = geo::distance(box.min_, box.max_);

  train_retriever tr{s, {}};
  auto const result = tr.trains(
      t_min, t_max, 100, 0,
      geo::make_box({{50.322747, 9.0133358}, {50.322747, 9.0133358}}), 18);

  EXPECT_EQ(4U, result.size());
  for (auto const& t : result) {
    EXPECT_FLOAT_EQ(expected_distance, t.route_distance_);

    auto const& dep = t.key_;
    EXPECT_TRUE(dep.is_not_null());
    EXPECT_TRUE(dep.is_departure());
    auto const arr = dep.get_opposite();

    EXPECT_GE(t_min, dep.get_time());
    EXPECT_LE(t_max, arr.get_time());
    EXPECT_LE(dep.get_time(), arr.get_time());

    auto const stations =
        std::make_pair(s.stations_.at(dep.get_station_idx())->eva_nr_,
                       s.stations_.at(arr.get_station_idx())->eva_nr_);

    EXPECT_TRUE(stations == std::make_pair(mcd::string{"5386096"},
                                           mcd::string{"7347220"}) ||
                stations == std::make_pair(mcd::string{"7347220"},
                                           mcd::string{"7190994"}));
  }
}
