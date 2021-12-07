#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>

#include "utl/to_set.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_stop.h"
#include "motis/loader/timezone_util.h"

#include "motis/core/schedule/validate_graph.h"

#include "./graph_builder_test.h"

using namespace motis::access;

namespace motis::loader {

using day_vec_t = std::vector<day_idx_t>;

class service_rules_day_shift_test : public loader_graph_builder_test {
public:
  service_rules_day_shift_test(std::string schedule_name,
                               std::string schedule_begin, int num_days)
      : loader_graph_builder_test(std::move(schedule_name),
                                  std::move(schedule_begin), num_days) {}

  std::vector<day_idx_t> trip_days(
      std::vector<station const*> const& stations) {
    std::vector<day_idx_t> days;
    for (auto const& trp : sched_->expanded_trips_.data_) {
      if (!check_trip_path(trp, stations)) {
        continue;
      }
      for (auto const& ctrp : trp->concrete_trips()) {
        days.push_back((*begin(stops(ctrp))).dep_time().day());
      }
    }
    std::sort(begin(days), end(days));
    return days;
  }

  //  static void check_trip_times(concrete_trip const trp) {
  //    auto last_time = 0U;
  //    for (auto const& section : sections(trp)) {
  //      auto const& lc = section.lcon();
  //      EXPECT_LE(last_time, lc.d_time_);
  //      EXPECT_LE(lc.d_time_, lc.a_time_);
  //      last_time = lc.a_time_;
  //    }
  //  }
};

class service_rules_day_shift_test_1 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_1()
      : service_rules_day_shift_test("mss-dayshift", "20180108", 3) {}
};

TEST_F(service_rules_day_shift_test_1, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    for (auto const& ctrp : trp->concrete_trips()) {
      //      check_trip_times(ctrp);
    }
  }
}

TEST_F(service_rules_day_shift_test_1, through_every_day) {
  auto const* a = get_station(*sched_, "0000001");
  auto const* c = get_station(*sched_, "0000003");
  auto const* d = get_station(*sched_, "0000004");
  auto const* e = get_station(*sched_, "0000005");
  auto const* g = get_station(*sched_, "0000007");
  auto const* h = get_station(*sched_, "0000008");
  auto const* i = get_station(*sched_, "0000009");
  auto const* j = get_station(*sched_, "0000010");
  auto const* k = get_station(*sched_, "0000011");

  for (auto const& t : sched_->expanded_trips_) {
    for (auto const& t1 : t) {
      for (auto const& ctrp : t1->concrete_trips()) {
        print_trip(ctrp);
      }
    }
  }

  //  EXPECT_EQ(num_days_ + 1, trip_count({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ(2, trip_count({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ(1, trip_count({a, c, d, e, g}));
  EXPECT_EQ(1, trip_count({g, h, i, j, k}));

  // days: 5 6 7

  EXPECT_EQ((day_vec_t{5, 6}), trip_days({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ((day_vec_t{7}), trip_days({a, c, d, e, g}));
  EXPECT_EQ((day_vec_t{5}), trip_days({g, h, i, j, k}));
}

TEST_F(service_rules_day_shift_test_1, through_one_day) {
  auto const* b = get_station(*sched_, "0000002");
  auto const* c = get_station(*sched_, "0000003");
  auto const* d = get_station(*sched_, "0000004");
  auto const* e = get_station(*sched_, "0000005");
  auto const* f = get_station(*sched_, "0000006");
  auto const* g = get_station(*sched_, "0000007");
  auto const* h = get_station(*sched_, "0000008");

  // days: 5 6 7

  EXPECT_EQ(1, trip_count({b, c, d, e, f, g, h}));
  EXPECT_EQ((day_vec_t{6}), trip_days({b, c, d, e, f, g, h}));
}

TEST_F(service_rules_day_shift_test_1, through_day_shift_between_services) {
  auto const* l = get_station(*sched_, "0000012");
  auto const* m = get_station(*sched_, "0000013");
  auto const* n = get_station(*sched_, "0000014");
  auto const* o = get_station(*sched_, "0000015");
  auto const* p = get_station(*sched_, "0000016");

  EXPECT_EQ(1, trip_count({l, m, n, o, p}));
  EXPECT_EQ((day_vec_t{6}), trip_days({l, m, n, o, p}));
}

class service_rules_day_shift_test_2 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_2()
      : service_rules_day_shift_test("mss-dayshift2", "20180108", 5) {}
};

TEST_F(service_rules_day_shift_test_2, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    for (auto const& ctrp : trp->concrete_trips()) {
      //      check_trip_times(ctrp);
    }
  }
}

TEST_F(service_rules_day_shift_test_2, expanded_trips) {
  print_graph(*sched_);

  auto const* a = get_station(*sched_, "0000001");
  auto const* b = get_station(*sched_, "0000002");
  auto const* c = get_station(*sched_, "0000003");
  auto const* d = get_station(*sched_, "0000004");
  auto const* e = get_station(*sched_, "0000005");
  auto const* f = get_station(*sched_, "0000006");
  auto const* g = get_station(*sched_, "0000007");
  auto const* h = get_station(*sched_, "0000008");
  auto const* i = get_station(*sched_, "0000009");
  auto const* j = get_station(*sched_, "0000010");
  auto const* k = get_station(*sched_, "0000011");

  // days: 5 6 7 8 9

  EXPECT_EQ((day_vec_t{5, 6, 7, 8, 9}), trip_days({a, b, c, d, e, g, k, j}));
  EXPECT_EQ((day_vec_t{5, 6}), trip_days({a, b, c, d, e, g, k, i, c, d, e}));
  EXPECT_EQ((day_vec_t{7, 8}), trip_days({a, b, c, d, e, g, k, i}));
  EXPECT_EQ((day_vec_t{6, 7, 8, 9}), trip_days({f, h, g, k, j}));
  EXPECT_EQ((day_vec_t{5, 6, 7}), trip_days({f, h, g, k, i, c, d, e}));
  EXPECT_EQ((day_vec_t{8, 9}), trip_days({f, h, g, k, i}));
  EXPECT_EQ((day_vec_t{5, 6}), trip_days({i, c, d, e}));
}

class service_rules_day_shift_test_3 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_3()
      : service_rules_day_shift_test("mss-dayshift3", "20180108", 5) {}
};

TEST_F(service_rules_day_shift_test_3, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    for (auto const& ctrp : trp->concrete_trips()) {
      //      check_trip_times(ctrp);
    }
  }
}

TEST_F(service_rules_day_shift_test_3, expanded_trips) {
  auto const* a = get_station(*sched_, "0000001");
  auto const* b = get_station(*sched_, "0000002");
  auto const* c = get_station(*sched_, "0000003");
  auto const* d = get_station(*sched_, "0000004");
  auto const* e = get_station(*sched_, "0000005");
  auto const* f = get_station(*sched_, "0000006");
  auto const* g = get_station(*sched_, "0000007");
  auto const* h = get_station(*sched_, "0000008");
  auto const* i = get_station(*sched_, "0000009");

  // days: 5 6 7 8 9

  EXPECT_EQ(1, trip_count({a, b, c, d, e, f}));
  EXPECT_EQ(1, trip_count({g, c, d, e, f}));
  EXPECT_EQ(1, trip_count({h, i, e, f}));
  EXPECT_EQ((day_vec_t{6}), trip_days({a, b, c, d, e, f}));
  EXPECT_EQ((day_vec_t{7}), trip_days({g, c, d, e, f}));
  EXPECT_EQ((day_vec_t{7}), trip_days({h, i, e, f}));
}

TEST_F(service_rules_day_shift_test_3, motis_trips) {
  EXPECT_EQ(3, sched_->trips_.size());
  EXPECT_EQ(std::set<uint32_t>({1, 2, 3}),
            utl::to_set(sched_->trips_, [](auto&& t) {
              return t.second->id_.primary_.train_nr_;
            }));
}

}  // namespace motis::loader
