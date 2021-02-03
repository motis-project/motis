#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>

#include "utl/to_set.h"
#include "utl/zip.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_stop.h"
#include "motis/loader/timezone_util.h"

#include "./graph_builder_test.h"

using namespace motis::access;

namespace motis::loader {

class service_rules_day_shift_test : public loader_graph_builder_test {
public:
  service_rules_day_shift_test(std::string schedule_name,
                               std::string schedule_begin, int num_days)
      : loader_graph_builder_test(std::move(schedule_name),
                                  std::move(schedule_begin), num_days) {}

  void print_trip(trip const* trp) {
    std::clog << "trip: ((" << trp->id_.primary_.station_id_ << ", "
              << trp->id_.primary_.train_nr_ << ", " << trp->id_.primary_.time_
              << "), (" << trp->id_.secondary_.target_station_id_ << ", "
              << trp->id_.secondary_.target_time_ << ", "
              << trp->id_.secondary_.line_id_ << "))" << std::endl;
    std::clog << "  " << trp->edges_->size()
              << " edges, lcon_idx=" << trp->lcon_idx_ << std::endl;
    std::clog << "  stops: ";
    for (auto const& stop : stops(trp)) {
      std::clog << stop.get_station(*sched_).name_ << " ";
    }
    std::clog << std::endl;
    for (auto const& sec : sections(trp)) {
      auto con_info = sec.lcon().full_con_->con_info_;
      std::clog << "  section " << sec.index() << ": "
                << sec.from_station(*sched_).name_ << " "
                << format_time(sec.lcon().d_time_) << " -> "
                << sec.to_station(*sched_).name_ << " "
                << format_time(sec.lcon().a_time_)
                << " train_nr=" << con_info->train_nr_;
      con_info = con_info->merged_with_;
      while (con_info != nullptr) {
        std::clog << " merged_with=" << con_info->train_nr_;
        con_info = con_info->merged_with_;
      }
      std::clog << std::endl;
    }
    std::clog << "\n\n";
  }

  int trip_count(std::vector<station const*> stations) {
    return static_cast<int>(std::count_if(
        begin(sched_->expanded_trips_.data_),
        end(sched_->expanded_trips_.data_),
        [&](trip const* trp) { return check_trip_path(trp, stations); }));
  }

  std::vector<int> trip_days(std::vector<station const*> stations) {
    std::vector<int> days;
    for (auto const& trp : sched_->expanded_trips_.data_) {
      if (check_trip_path(trp, stations)) {
        days.push_back((*stops(trp).begin()).dep_lcon().d_time_ / 1440);
      }
    }
    std::sort(begin(days), end(days));
    return days;
  }

  bool check_trip_path(trip const* trp, std::vector<station const*>& stations) {
    auto const stps = stops(trp);
    auto const trip_stops = std::vector<trip_stop>(begin(stps), end(stps));
    if (trip_stops.size() != stations.size()) {
      return false;
    }
    // NOLINTNEXTLINE
    for (auto const& [stop, station] : utl::zip(trip_stops, stations)) {
      if (&stop.get_station(*sched_) != station) {
        return false;
      }
    }
    return true;
  }

  static void check_trip_times(trip const* trp) {
    auto last_time = 0U;
    for (auto const& section : sections(trp)) {
      auto const& lc = section.lcon();
      EXPECT_LE(last_time, lc.d_time_);
      EXPECT_LE(lc.d_time_, lc.a_time_);
      last_time = lc.a_time_;
    }
  }
};

class service_rules_day_shift_test_1 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_1()
      : service_rules_day_shift_test("mss-dayshift", "20180108", 3) {}
};

TEST_F(service_rules_day_shift_test_1, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    check_trip_times(trp);
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

  //  EXPECT_EQ(num_days_ + 1, trip_count({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ(2, trip_count({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ(1, trip_count({a, c, d, e, g}));
  EXPECT_EQ(1, trip_count({g, h, i, j, k}));

  // days: 5 6 7

  EXPECT_EQ((std::vector<int>{5, 6}), trip_days({a, c, d, e, g, h, i, j, k}));
  EXPECT_EQ((std::vector<int>{7}), trip_days({a, c, d, e, g}));
  EXPECT_EQ((std::vector<int>{5}), trip_days({g, h, i, j, k}));
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
  EXPECT_EQ((std::vector<int>{6}), trip_days({b, c, d, e, f, g, h}));
}

TEST_F(service_rules_day_shift_test_1, through_day_shift_between_services) {
  auto const* l = get_station(*sched_, "0000012");
  auto const* m = get_station(*sched_, "0000013");
  auto const* n = get_station(*sched_, "0000014");
  auto const* o = get_station(*sched_, "0000015");
  auto const* p = get_station(*sched_, "0000016");

  EXPECT_EQ(1, trip_count({l, m, n, o, p}));
  EXPECT_EQ((std::vector<int>{6}), trip_days({l, m, n, o, p}));
}

class service_rules_day_shift_test_2 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_2()
      : service_rules_day_shift_test("mss-dayshift2", "20180108", 5) {}
};

TEST_F(service_rules_day_shift_test_2, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    check_trip_times(trp);
  }
}

TEST_F(service_rules_day_shift_test_2, expanded_trips) {
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

  EXPECT_EQ((std::vector<int>{5, 6, 7, 8, 9}),
            trip_days({a, b, c, d, e, g, k, j}));
  EXPECT_EQ((std::vector<int>{5, 6}),
            trip_days({a, b, c, d, e, g, k, i, c, d, e}));
  EXPECT_EQ((std::vector<int>{7, 8}), trip_days({a, b, c, d, e, g, k, i}));
  EXPECT_EQ((std::vector<int>{6, 7, 8, 9}), trip_days({f, h, g, k, j}));
  EXPECT_EQ((std::vector<int>{5, 6, 7}), trip_days({f, h, g, k, i, c, d, e}));
  EXPECT_EQ((std::vector<int>{8, 9}), trip_days({f, h, g, k, i}));
  EXPECT_EQ((std::vector<int>{5, 6}), trip_days({i, c, d, e}));
}

class service_rules_day_shift_test_3 : public service_rules_day_shift_test {
public:
  service_rules_day_shift_test_3()
      : service_rules_day_shift_test("mss-dayshift3", "20180108", 5) {}
};

TEST_F(service_rules_day_shift_test_3, valid_trip_times) {
  for (auto const& trp : sched_->expanded_trips_.data_) {
    check_trip_times(trp);
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
  EXPECT_EQ((std::vector<int>{6}), trip_days({a, b, c, d, e, f}));
  EXPECT_EQ((std::vector<int>{7}), trip_days({g, c, d, e, f}));
  EXPECT_EQ((std::vector<int>{7}), trip_days({h, i, e, f}));
}

TEST_F(service_rules_day_shift_test_3, motis_trips) {
  EXPECT_EQ(3, sched_->trips_.size());
  EXPECT_EQ(std::set<uint32_t>({1, 2, 3}),
            utl::to_set(sched_->trips_, [](auto&& t) {
              return t.second->id_.primary_.get_train_nr();
            }));
}

}  // namespace motis::loader
