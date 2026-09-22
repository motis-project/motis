#include "gtest/gtest.h"

#include <filesystem>
#include <optional>

#include "utl/init_from.h"

#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace motis;
namespace n = nigiri;

namespace {

// All stops lie far outside of the OSM extract, so the street router connects
// none of them: what the default profile holds for a pair is a rule or an
// estimate.
//
//   P1 -- 300m -- P2   platforms of station P
//   P1 --  78m -- Q    another stop, closer than 100m
//   P1 -- 400m -- R    another stop
//   P2 -> R            stated by transfers.txt (11 min)
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
X,X,60.1000,5.0000,0,
P,P,60.0000,5.0000,1,
P1,P1,60.0000,5.0000,0,P
P2,P2,60.0027,5.0000,0,P
Q,Q,59.9993,5.0000,0,
R,R,59.9964,5.0000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,X,0,0,0
T1,10:30:00,10:30:00,P1,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
P2,R,2,660

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

data import_with(bool const osr_footpath, char const* dir) {
  auto const path = std::filesystem::path{"test/data"} / dir;
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);
  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = osr_footpath};
  import(c, path);
  return data{path, c};
}

// What the routing takes for a -> b: a footpath or what a hub hands out.
std::optional<int> minutes(n::timetable const& tt,
                           n::profile_idx_t const prf,
                           char const* from,
                           char const* to) {
  auto const src = n::source_idx_t{0U};
  auto const a = tt.locations_.location_id_to_idx_.at({from, src});
  auto const b = tt.locations_.location_id_to_idx_.at({to, src});
  auto best = std::optional<int>{};
  auto const take = [&](n::location_idx_t const l, n::duration_t const d) {
    if (l == b && (!best.has_value() || d.count() < *best)) {
      best = d.count();
    }
  };
  if (cista::to_idx(a) < tt.locations_.footpaths_out_[prf].size()) {
    for (auto const fp : tt.locations_.footpaths_out_[prf][a]) {
      take(fp.target(), fp.duration());
    }
  }
  n::routing::for_each_hub_source<n::direction::kBackward>(
      tt, prf, a, [&](n::footpath const fp) {
        take(fp.target(), fp.duration());
        return true;
      });
  return best;
}

}  // namespace

// osr_footpath is the switch: without it the default profile is the loader's.
TEST(motis, default_profile_walks_without_osr_footpath) {
  auto const d = import_with(false, "default_profile_beeline");
  auto const& tt = *d.tt_;
  EXPECT_EQ(4, minutes(tt, n::kDefaultProfile, "P1", "P2"));  // 300m / 1.5m/s
  EXPECT_EQ(2, minutes(tt, n::kDefaultProfile, "P1", "Q"));
  EXPECT_EQ(std::nullopt, minutes(tt, n::kDefaultProfile, "P1", "R"));
  EXPECT_EQ(11, minutes(tt, n::kDefaultProfile, "P2", "R"));
}

// With it, the default profile walks where the router walks. A pair the router
// cannot connect keeps an estimate (beeline at 0.7m/s) if it is closer than
// 100m or within one station - and a rule stays a rule.
TEST(motis, default_profile_walks_with_osr_footpath) {
  auto const d = import_with(true, "default_profile_routed");
  auto const& tt = *d.tt_;
  EXPECT_EQ(8, minutes(tt, n::kDefaultProfile, "P1", "P2"));  // one station
  EXPECT_EQ(8, minutes(tt, n::kDefaultProfile, "P2", "P1"));
  EXPECT_EQ(2, minutes(tt, n::kDefaultProfile, "P1", "Q"));  // < 100m
  EXPECT_EQ(std::nullopt, minutes(tt, n::kDefaultProfile, "P1", "R"));
  EXPECT_EQ(11, minutes(tt, n::kDefaultProfile, "P2", "R"));  // transfers.txt

  // the foot profile stays physical: extend_missing_footpaths is off
  EXPECT_EQ(std::nullopt, minutes(tt, n::kFootProfile, "P1", "P2"));
  EXPECT_EQ(std::nullopt, minutes(tt, n::kFootProfile, "P1", "Q"));
  EXPECT_EQ(std::nullopt, minutes(tt, n::kFootProfile, "P2", "R"));
}

namespace {

// RE2 arrives at FFM_10 10:25, S3a leaves FFM_101 10:30, S3b 10:40. The beeline
// between the two platforms takes 3 min, the routed walk 6 min.
constexpr auto const kPlatformChangeGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
LANGEN,Langen,49.99359,8.65677,1,,
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RE2,DB,RE2,,,106
S3,DB,S3,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RE2,S1,RE2,,
S3,S1,S3a,,
S3,S1,S3b,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
RE2,10:10:00,10:10:00,LANGEN,0,0,0
RE2,10:25:00,10:25:00,FFM_10,1,0,0
S3a,10:30:00,10:30:00,FFM_101,0,0,0
S3a,10:38:00,10:38:00,FFM_HAUPT_S,1,0,0
S3b,10:40:00,10:40:00,FFM_101,0,0,0
S3b,10:48:00,10:48:00,FFM_HAUPT_S,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

}  // namespace

// The trip-based transfers are precomputed at import time: with osr_footpath
// they have to come from the timetable the server routes on (tt_ext.bin), or
// they would still allow the 3 min beeline change onto S3a.
TEST(motis, trip_based_transfers_follow_osr_footpath) {
  auto const path = std::filesystem::path{"test/data/default_profile_tb"};
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);
  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{
                     .first_day_ = "2019-05-01",
                     .num_days_ = 2,
                     .tb_ = true,
                     .datasets_ = {{"test", {.path_ = kPlatformChangeGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true};
  import(c, path);
  auto d = data{path, c};
  ASSERT_NE(nullptr, d.tbd_);

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = [&](char const* algorithm) {
    return routing(std::string{"?fromPlace=test_LANGEN&toPlace=test_FFM_HAUPT_S"
                               "&time=2019-05-01T08:05:00Z&algorithm="} +
                   algorithm);
  };
  auto const raptor = plan("RAPTOR");
  auto const tb = plan("TB");
  EXPECT_FALSE(raptor.debugOutput_.contains("n_segments_enqueued"));
  ASSERT_TRUE(tb.debugOutput_.contains("n_segments_enqueued"));  // TB ran

  for (auto const* res : {&raptor, &tb}) {
    ASSERT_FALSE(res->itineraries_.empty());
    EXPECT_EQ("2019-05-01 08:48",  // S3b
              date::format("%F %H:%M", *res->itineraries_.front().endTime_));
  }
}
