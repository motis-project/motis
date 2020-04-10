#include "gtest/gtest.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_section.h"
#include "motis/core/access/trip_stop.h"

#include "./graph_builder_test.h"

using namespace motis::access;

namespace motis::loader {

class loader_trip : public loader_graph_builder_test {
public:
  loader_trip() : loader_graph_builder_test("trip", "20151025", 2) {}
};

TEST_F(loader_trip, none) {
  ASSERT_ANY_THROW(get_trip(*sched_, "1234567", 31337, unix_time(100),
                            "7654321", unix_time(200), ""));
}

TEST_F(loader_trip, simple) {
  auto trp = get_trip(*sched_, "0000001", 1, unix_time(1000), "0000003",
                      unix_time(1200), "");
  ASSERT_NE(nullptr, trp);

  auto const& primary = trp->id_.primary_;
  auto const& secondary = trp->id_.secondary_;

  EXPECT_EQ("0000001", sched_->stations_[primary.station_id_]->eva_nr_);
  EXPECT_EQ(1, primary.train_nr_);
  EXPECT_EQ(motis_time(1000), primary.time_);

  EXPECT_EQ("", secondary.line_id_);
  EXPECT_EQ("0000003",
            sched_->stations_[secondary.target_station_id_]->eva_nr_);
  EXPECT_EQ(motis_time(1200), secondary.target_time_);

  ASSERT_EQ(2, trp->edges_->size());
  for (auto const& sec : sections(trp)) {
    auto const& lcon = sec.lcon();
    auto const& info = sec.info(*sched_);
    auto const& from = sec.from_station(*sched_);
    auto const& to = sec.to_station(*sched_);

    switch (sec.index()) {
      case 0:
        EXPECT_EQ(motis_time(1000), lcon.d_time_);
        EXPECT_EQ(motis_time(1100), lcon.a_time_);
        EXPECT_EQ("0000001", from.eva_nr_);
        EXPECT_EQ("0000002", to.eva_nr_);
        EXPECT_EQ(1, info.train_nr_);
        break;

      case 1:
        EXPECT_EQ(motis_time(1100), lcon.d_time_);
        EXPECT_EQ(motis_time(1200), lcon.a_time_);
        EXPECT_EQ("0000002", from.eva_nr_);
        EXPECT_EQ("0000003", to.eva_nr_);
        EXPECT_EQ(1, info.train_nr_);
        break;

      default: FAIL() << "section index out of bounds";
    }
  }

  for (auto const& stop : stops(trp)) {
    auto const& station = stop.get_station(*sched_);
    switch (stop.index()) {
      case 0:
        EXPECT_EQ("0000001", station.eva_nr_);
        ASSERT_FALSE(stop.has_arrival());
        ASSERT_TRUE(stop.has_departure());
        EXPECT_EQ(motis_time(1000), stop.dep_lcon().d_time_);

        break;

      case 1:
        EXPECT_EQ("0000002", station.eva_nr_);
        ASSERT_TRUE(stop.has_arrival());
        ASSERT_TRUE(stop.has_departure());
        EXPECT_EQ(motis_time(1100), stop.arr_lcon().a_time_);
        EXPECT_EQ(motis_time(1100), stop.dep_lcon().d_time_);
        break;

      case 2:
        EXPECT_EQ("0000003", station.eva_nr_);
        ASSERT_TRUE(stop.has_arrival());
        ASSERT_FALSE(stop.has_departure());
        EXPECT_EQ(motis_time(1200), stop.arr_lcon().a_time_);
        break;

      default: FAIL() << "stop index out of bounds";
    }
  }
}

TEST_F(loader_trip, collision) {
  auto trp0 = get_trip(*sched_, "0000004", 2, unix_time(1000), "0000005",
                       unix_time(1100), "foo");
  auto trp1 = get_trip(*sched_, "0000004", 2, unix_time(1000), "0000005",
                       unix_time(1100), "bar");

  ASSERT_NE(nullptr, trp0);
  ASSERT_NE(nullptr, trp1);
  ASSERT_NE(trp0, trp1);
}

TEST_F(loader_trip, rename) {
  auto trp0 = get_trip(*sched_, "0000001", 3, unix_time(2000), "0000003",
                       unix_time(2200), "");
  auto trp1 = get_trip(*sched_, "0000002", 4, unix_time(2100), "0000003",
                       unix_time(2200), "");

  ASSERT_NE(nullptr, trp0);
  ASSERT_NE(nullptr, trp1);
  ASSERT_EQ(trp0, trp1);
}

}  // namespace motis::loader
