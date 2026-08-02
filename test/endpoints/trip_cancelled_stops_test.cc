#include "gtest/gtest.h"
#include "motis/endpoints/stop_times.h"

#include <chrono>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"

#include "../util.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;
namespace n = nigiri;

// Three ICE trips connected via `block_id` into a single vehicle run:
//
//   T_PREV:  P -> S0
//   T_MID:        S0 -> S1 -> S2 -> S3 -> S4
//   T_NEXT:                               S4 -> Q
//
// The connecting stops (S0 between T_PREV/T_MID, S4 between T_MID/T_NEXT) are
// shared, so the merged run is P, S0, S1, S2, S3, S4, Q. The *middle* trip
// T_MID is the one that gets updated/cancelled. This way the tests exercise
// that `get_first_trip_stop` / `get_last_trip_stop` (used by `tripFrom` /
// `tripTo`) stop at the boundaries of T_MID and never bleed into the
// neighbouring trips, even when boundary stops are cancelled.
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
P,Stop P,49.90,8.00,1,
S0,Stop 0,50.00,8.00,1,
S1,Stop 1,50.10,8.00,1,
S2,Stop 2,50.20,8.00,1,
S3,Stop 3,50.30,8.00,1,
S4,Stop 4,50.40,8.00,1,
Q,Stop Q,50.50,8.00,1,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,SVC,T_PREV,,BLK
ICE,SVC,T_MID,,BLK
ICE,SVC,T_NEXT,,BLK

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_PREV,09:00:00,09:00:00,P,0,0,0
T_PREV,09:30:00,09:30:00,S0,1,0,0
T_MID,09:30:00,09:30:00,S0,0,0,0
T_MID,09:40:00,09:40:00,S1,1,0,0
T_MID,09:50:00,09:50:00,S2,2,0,0
T_MID,10:00:00,10:00:00,S3,3,0,0
T_MID,10:10:00,10:10:00,S4,4,0,0
T_NEXT,10:10:00,10:10:00,S4,0,0,0
T_NEXT,10:30:00,10:30:00,Q,1,0,0

# calendar_dates.txt
service_id,date,exception_type
SVC,20190501,1
)";

namespace {

data import_gtfs(char const* data_dir) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(data_dir, ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, data_dir);
  auto d = data{data_dir, c};
  d.init_rtt(date::sys_days{2019_y / May / 1});
  return d;
}

// Queries the `/stoptimes` endpoint at S2 (arrival) and returns the stop time.
// `with_skipped` includes stops that are scheduled but skipped by a real-time
// update (otherwise they are omitted from the response). `trip_id_suffix`
// selects the matching run when S2 is served by more than one trip (e.g. the
// scheduled trip plus an additional trip); empty matches the first stop time.
api::StopTime s2_stoptime(ep::stop_times const& stop_times,
                          std::string_view const trip_id_suffix = {},
                          bool const with_skipped = false) {
  auto const url =
      std::string{
          "/api/v5/stoptimes?stopId=test_S2"
          "&time=2019-05-01T08:30:00.000Z"
          "&arriveBy=true"
          "&n=5"
          "&language=de"
          "&fetchStops=true"} +
      (with_skipped ? "&withScheduledSkippedStops=true" : "");
  auto const res = stop_times(url);
  EXPECT_EQ("test_S2", res.place_.stopId_);
  for (auto const& st : res.stopTimes_) {
    if (trip_id_suffix.empty() || st.tripId_.ends_with(trip_id_suffix)) {
      return st;
    }
  }
  ADD_FAILURE() << "no stop time at S2 matching trip id suffix \""
                << trip_id_suffix << "\"";
  return {};
}

}  // namespace

TEST(motis, trip_cancelled_first_and_last_stop) {
  auto d = import_gtfs("test/data_trip_cancelled");
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  // Without any real-time update, tripFrom/tripTo are the first/last stops of
  // the *middle* trip T_MID (S0 / S4), not of the whole block run (P / Q).
  {
    auto const ice = s2_stoptime(stop_times);
    EXPECT_EQ("test_S0", ice.tripFrom_.stopId_);
    EXPECT_EQ("test_S4", ice.tripTo_.stopId_);
  }

  // Skip the first (S0) and the last (S4) stop of the middle trip T_MID.
  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T_MID", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "S0", .skip_ = true},
                                         {.stop_id_ = "S4", .skip_ = true}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  // Now the first served stop of T_MID is S1 and the last served stop is S3.
  {
    auto const ice = s2_stoptime(stop_times);
    EXPECT_EQ("test_S1", ice.tripFrom_.stopId_);
    EXPECT_EQ("test_S3", ice.tripTo_.stopId_);
  }
}

TEST(motis, trip_all_stops_cancelled) {
  auto d = import_gtfs("test/data_trip_all_cancelled");
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  // Skip every stop of the middle trip T_MID (S0 .. S4).
  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T_MID", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "S0", .skip_ = true},
                                         {.stop_id_ = "S1", .skip_ = true},
                                         {.stop_id_ = "S2", .skip_ = true},
                                         {.stop_id_ = "S3", .skip_ = true},
                                         {.stop_id_ = "S4", .skip_ = true}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  // S2 itself is now skipped, so it only shows up with `withScheduledSkipped`.
  // With every stop of T_MID cancelled the boundary search must not run out of
  // the trip into the neighbouring trips: it stops at the last (S4) / first
  // (S0) stop *of T_MID* and never reaches T_NEXT / Q or T_PREV / P.
  auto const ice = s2_stoptime(stop_times, /*trip_id_suffix=*/{},
                               /*with_skipped=*/true);
  EXPECT_TRUE(ice.cancelled_);
  EXPECT_EQ("test_S4", ice.tripFrom_.stopId_);
  EXPECT_EQ("test_S0", ice.tripTo_.stopId_);
}

TEST(motis, trip_added_cancelled_first_and_last_stop) {
  auto d = import_gtfs("test/data_trip_added_cancelled");
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  // Create an additional (not scheduled) trip "ADDED_ICE" serving S0 .. S4
  // and immediately skip the first (S0) and last (S4) stop. For additional
  // trips the stops are taken from the update's absolute times, so every stop
  // (including the skipped boundary ones) needs a time.
  auto const t = [](int const m) {
    return date::sys_seconds{date::sys_days{2019_y / May / 1} + 8h +
                             std::chrono::minutes{m}};
  };
  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "ADDED_ICE", .route_id_ = {"ICE"}},
                       .stop_updates_ =
                           {{.stop_id_ = "S0", .skip_ = true, .time_ = t(0)},
                            {.stop_id_ = "S1", .time_ = t(10)},
                            {.stop_id_ = "S2", .time_ = t(20)},
                            {.stop_id_ = "S3", .time_ = t(30)},
                            {.stop_id_ = "S4", .skip_ = true, .time_ = t(40)}},
                       .added_ = true}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  // S2 is served by both the scheduled T_MID and the additional ADDED_ICE, so
  // select the additional trip by its id. For an additional trip the boundary
  // search walks over the skipped boundary stops: the first served stop is S1
  // and the last served stop is S3.
  auto const ice = s2_stoptime(stop_times, /*trip_id_suffix=*/"ADDED_ICE");
  EXPECT_TRUE(ice.realTime_);
  EXPECT_EQ("test_S1", ice.tripFrom_.stopId_);
  EXPECT_EQ("test_S3", ice.tripTo_.stopId_);
}
