#include "gtest/gtest.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

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

class expand_trips_test : public loader_graph_builder_test {
public:
  expand_trips_test() : loader_graph_builder_test("mss-ts", "20150325", 3) {}

  int trip_count(std::vector<station const*> stations) {
    return static_cast<int>(std::count_if(
        begin(sched_->expanded_trips_.data_),
        end(sched_->expanded_trips_.data_), [&](concrete_trip const trp) {
          return check_trip_path(trp, stations);
        }));
  }

  bool check_trip_path(concrete_trip const trp,
                       std::vector<station const*>& stations) {
    auto const stps = stops(trp);
    auto const trip_stops = std::vector<trip_stop>(begin(stps), end(stps));
    if (trip_stops.size() != stations.size()) {
      return false;
    }
    // NOLINTNEXTLINE(readability-use-anyofallof)
    for (auto const& [stop, station] : utl::zip(trip_stops, stations)) {
      if (&stop.get_station(*sched_) != station) {
        return false;
      }
    }
    return true;
  }
};

TEST_F(expand_trips_test, check_expanded_trips) {
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
  EXPECT_EQ(6, sched_->expanded_trips_.index_size() - 1);
  EXPECT_EQ(6 * num_days_, sched_->expanded_trips_.data_size());
  EXPECT_EQ(num_days_, trip_count({b, c, d, f, h, j, k}));
  EXPECT_EQ(num_days_, trip_count({b, c, d, e, g, i}));
  EXPECT_EQ(num_days_, trip_count({b, c, d, e, j, k}));
  EXPECT_EQ(num_days_, trip_count({a, c, d, f, h, j, k}));
  EXPECT_EQ(num_days_, trip_count({a, c, d, e, g, i}));
  EXPECT_EQ(num_days_, trip_count({a, c, d, e, j, k}));
}

}  // namespace motis::loader
