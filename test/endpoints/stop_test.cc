#include "motis/endpoints/stop.h"
#include "gtest/gtest.h"

#include <set>

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"

using namespace motis;
using namespace date;
namespace n = nigiri;

// Routes:
//   R1: A -> B -> C            (regular)
//   R2: B -> C -> D            (regular, shares B->C with R1)
//   R3: A -> B -> C            (same stop sequence as R1: tests
//   same-seq/diff-route-id fix) RP: X -> Y  (block_id=blk) (block-merged: tests
//   correct stop position) RQ: Y -> Z  (block_id=blk) (block-merged: continues
//   from RP)
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,49.00,8.00
B,B,49.10,8.00
C,C,49.20,8.00
D,D,49.30,8.00
X,X,50.00,9.00
Y,Y,50.10,9.00
Z,Z,50.20,9.00

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R1,DB,R1,,3
R2,DB,R2,,3
R3,DB,R3,,3
RP,DB,RP,,3
RQ,DB,RQ,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R3,S1,T3,,
RP,S1,TP,,blk
RQ,S1,TQ,,blk

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,01:00:00,01:00:00,A,0
T1,01:10:00,01:10:00,B,1
T1,01:20:00,01:20:00,C,2
T2,01:00:00,01:00:00,B,0
T2,01:10:00,01:10:00,C,1
T2,01:20:00,01:20:00,D,2
T3,01:00:00,01:00:00,A,0
T3,01:10:00,01:10:00,B,1
T3,01:20:00,01:20:00,C,2
TP,02:00:00,02:00:00,X,0
TP,02:10:00,02:10:00,Y,1
TQ,02:10:00,02:10:00,Y,0
TQ,02:20:00,02:20:00,Z,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, stop) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data/stop", ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, "test/data/stop");
  auto d = data{"test/data/stop", c};

  auto const ep = utl::init_from<ep::stop>(d).value();

  auto const route_ids_at = [&](char const* stop_id) {
    auto ids = std::set<std::string>{};
    for (auto const& r :
         ep(std::string{"/api/v6/stop?stopId="} + stop_id).routes_) {
      ids.insert(r.routeId_);
    }
    return ids;
  };

  // Stop A: only R1 and R3 depart (same stop sequence A->B->C)
  EXPECT_EQ((std::set<std::string>{"test_R1", "test_R3"}),
            route_ids_at("test_A"))
      << "fix: both routes sharing the same stop sequence must be returned";

  // Stop B: R1, R2, and R3 all pass through B
  EXPECT_EQ((std::set<std::string>{"test_R1", "test_R2", "test_R3"}),
            route_ids_at("test_B"));

  // Stop C: R1, R2, and R3 — R1/R3 terminate here (kArr), R2 passes through
  {
    auto const ids = route_ids_at("test_C");
    EXPECT_TRUE(ids.count("test_R1"));
    EXPECT_TRUE(ids.count("test_R2"));
    EXPECT_TRUE(ids.count("test_R3"));
  }

  // Stop D: only R2 terminates here
  EXPECT_EQ((std::set<std::string>{"test_R2"}), route_ids_at("test_D"));

  // Block-merged transport: RP (X->Y) + RQ (Y->Z) merged via block_id=blk
  // into a single transport X->Y->Z.

  // Stop X (first stop of merged transport): route is RP
  EXPECT_EQ((std::set<std::string>{"test_RP"}), route_ids_at("test_X"))
      << "fix: block-merged transport must use correct stop position (not stop "
         "0)";

  // Stop Z (last stop of merged transport): route is RQ
  EXPECT_EQ((std::set<std::string>{"test_RQ"}), route_ids_at("test_Z"))
      << "fix: last stop of block-merged transport must use kArr to avoid "
         "out-of-bounds section access";

  // place_ must be filled for a stopId query
  auto const resp = ep("/api/v6/stop?stopId=test_A");
  EXPECT_FALSE(resp.place_.name_.empty());
}
