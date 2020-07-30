#include "gtest/gtest.h"

#include "motis/test/motis_instance_test.h"
#include "../hrd/paths.h"

#include "motis/core/access/station_access.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::loader;

struct multischedule_dedup_test : public motis_instance_test {
  multischedule_dedup_test()
      : motis_instance_test(
            loader_options{
                .dataset_ =
                    {(hrd::SCHEDULES / "single-ice").generic_string(),
                     (hrd::SCHEDULES / "multiple-ices").generic_string()},
                .schedule_begin_ = "20151004",
                .dataset_prefix_ = {{"s"}, {"m"}}},
            {"routing"}) {}
};

TEST_F(multischedule_dedup_test, stations) {
  {  // some stations from "single-ice"
    EXPECT_NE(nullptr, find_station(sched(), "s_8000261"));  // München Hbf
    EXPECT_NE(nullptr, find_station(sched(), "s_8011102"));  // Gesundbrunnen
  }
  {  // some stations from "multiple-ices"
    EXPECT_NE(nullptr, find_station(sched(), "m_8000261"));  // München Hbf
    EXPECT_NE(nullptr, find_station(sched(), "m_8011102"));  // Gesundbrunnen
    EXPECT_NE(nullptr, find_station(sched(), "m_8010222"));  // Wittenberg
  }
}

TEST_F(multischedule_dedup_test, equivalents) {
  auto const* ss = find_station(sched(), "s_8000261");
  auto const* sm = find_station(sched(), "m_8000261");

  EXPECT_TRUE(
      std::any_of(begin(ss->equivalent_), end(ss->equivalent_),
                  [&](auto const& s) { return s->index_ == sm->index_; }));

  EXPECT_TRUE(
      std::any_of(begin(sm->equivalent_), end(sm->equivalent_),
                  [&](auto const& s) { return s->index_ == ss->index_; }));
}

TEST_F(multischedule_dedup_test, trips) { EXPECT_EQ(1, sched().trips_.size()); }
