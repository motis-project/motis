#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/nigiri/metrics.h"
#include "motis/nigiri/routing.h"
#include "motis/nigiri/tag_lookup.h"

#include "./utils.h"

using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
namespace n = nigiri;
namespace mn = motis::nigiri;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00, 1h, 1 transfer
// [TRIP 1: A, B, C] -> WALK [5 min] -> [TRIP 2: D, E]
//
// 10:00 - 10:45, 45min, 2 transfers
// [TRIP 1: A, B] -> [TRIP 3: B, C] -> [TRIP 4: C, E]
//
// TIMETABLE + DELAYS:
// TRIP  |  STOP  |  EV  |  PLANNED  | REAL TIME     | SPECIAL
// ------+--------+------+-----------+---------------+---------------
// 0     |  A     | DEP  |  09:50    |               | TRIP CANCEL
// 0     |  E     | ARR  |  10:00    |               | TRIP CANCEL
// 1     |  A     | DEP  |  09:50    | 10:00 (+10)   |
// 1     |  E     | DEP  |  09:55    |               | STOP SKIP
// 1     |  E     | ARR  |  09:55    |               | STOP SKIP
// 1     |  B     | ARR  |  10:00    | 10:10 (+10)   |
// 1     |  B     | DEP  |  10:10    | 10:15 (+5)    |
// 1     | C -> C1| ARR  |  10:30    | 10:35 (+5)    | STOP ASSIGNMENT CHANGE
// 2     |  D     | DEP  |  10:25    | 10:40 (+15)   |
// 2     |  E     | ARR  |  10:45    | 11:00 (+15)   |
// 3     |  B     | DEP  |  10:10    | 10:20 (+10)   |
// 3     |  C     | ARR  |  10:30    | 10:30 (+0)    |
// 4     |  C     | DEP  |  10:30    | 10:35 (+5)    |
// 4     |  E     | ARR  |  10:45    | 10:45 (+0)    |
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
C1,C1,,5.0,4.0,,
D,D,,6.0,7.0,,
E,E,,7.0,8.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,X,,,3
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,TX,,
R1,S1,T1,,
R2,S1,T2,,
R3,S1,T3,,
R4,S1,T4,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TX,09:50:00,09:50:00,A,0,0,0
TX,10:00:00,10:00:00,E,1,0,0
T1,09:50:00,09:50:00,A,0,0,0
T1,09:55:00,09:55:00,E,1,0,0
T1,10:00:00,10:10:00,B,2,0,0
T1,10:30:00,10:30:00,C,3,0,0
T2,10:25:00,10:25:00,D,1,0,0
T2,10:45:00,10:45:00,E,2,0,0
T3,10:10:00,10:10:00,B,1,0,0
T3,10:30:00,10:30:00,C,2,0,0
T4,10:30:00,10:30:00,C,1,0,0
T4,10:45:00,10:45:00,E,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
C1,D,0,2
)"sv;

constexpr auto const expected =
    R"(Journey: duration=60  transfers=1  accessibility=0                01.05. 07:50 +10 F --> 01.05. 08:45 +15 F (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 07:50 +10 F       enter
 1: tag_B   B............................................... a: 01.05. 08:00 +10 F  d: 01.05. 08:10 +5  F
 2: tag_C1  C1.............................................. a: 01.05. 08:30 +5  F  d: 01.05. 08:30 +5  F  exit
 3: tag_D   D............................................... a: 01.05. 08:32 +5  F  d: 01.05. 08:25 +15 F       enter
 4: tag_E   E............................................... a: 01.05. 08:45 +15 F  d:                     exit

Transports:
 0: 0  -> 2  TRAIN Bus 1                     duration=35, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="35"
 1: 2  -> 3  WALK  type=          id=-1      duration=2   accessibility=0
 2: 3  -> 4  TRAIN Bus 2                     duration=20, provider="Deutsche Bahn", direction="", line="2", clasz=3, duration="20"

Trips:
 0: 0  -> 2  {tag_A  ,      0, 2019-05-01 07:50} -> {  tag_C, 2019-05-01 08:30}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697000/tag_C/1556699400/1  ::memory::/stop_times.txt:4:7
 1: 3  -> 4  {tag_D  ,      0, 2019-05-01 08:25} -> {  tag_E, 2019-05-01 08:45}, line_id=2, id=tag_T2
       #/trip/tag_D/0/1556699100/tag_E/1556700300/2  ::memory::/stop_times.txt:8:9

Attributes:
Journey: duration=45  transfers=2  accessibility=0                01.05. 07:50 +10 F --> 01.05. 08:45 +0  F (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 07:50 +10 F       enter
 1: tag_B   B............................................... a: 01.05. 08:00 +10 F  d: 01.05. 08:10 +10 F  exit enter
 2: tag_C   C............................................... a: 01.05. 08:30 +0  F  d: 01.05. 08:30 +5  F  exit enter
 3: tag_E   E............................................... a: 01.05. 08:45 +0  F  d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 1                     duration=10, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="10"
 1: 1  -> 2  TRAIN Bus 3                     duration=10, provider="Deutsche Bahn", direction="", line="3", clasz=3, duration="10"
 2: 2  -> 3  TRAIN Bus 4                     duration=10, provider="Deutsche Bahn", direction="", line="4", clasz=3, duration="10"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 07:50} -> {  tag_C, 2019-05-01 08:30}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697000/tag_C/1556699400/1  ::memory::/stop_times.txt:4:7
 1: 1  -> 2  {tag_B  ,      0, 2019-05-01 08:10} -> {  tag_C, 2019-05-01 08:30}, line_id=3, id=tag_T3
       #/trip/tag_B/0/1556698200/tag_C/1556699400/3  ::memory::/stop_times.txt:10:11
 2: 2  -> 3  {tag_C  ,      0, 2019-05-01 08:30} -> {  tag_E, 2019-05-01 08:45}, line_id=4, id=tag_T4
       #/trip/tag_C/0/1556699400/tag_E/1556700300/4  ::memory::/stop_times.txt:12:13

Attributes:
)";

}  // namespace

TEST(nigiri, rt_test) {
  using namespace motis;
  using motis::routing::RoutingResponse;

  auto tt = n::timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  n::loader::register_special_stations(tt);
  n::loader::gtfs::load_timetable({}, n::source_idx_t{0},
                                  n::loader::mem_dir::read(test_files), tt);
  n::loader::finalize(tt);

  auto rtt = n::rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});

  auto tags = mn::tag_lookup{};
  tags.add(n::source_idx_t{0U}, "tag_");

  auto prometheus_registry = prometheus::Registry{};
  auto metrics = mn::metrics{prometheus_registry};

  /*** BASE LINE:  A@09:00 -> E@09:55 direct via T1 ***/
  auto const r0 =
      mn::route(tags, tt, &rtt,
                mn::make_routing_msg(
                    "tag_A", "tag_E",
                    mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min)),
                metrics);
  auto const j0 = message_to_journeys(motis_content(RoutingResponse, r0));
  ASSERT_EQ(1U, j0.size());
  EXPECT_EQ(0U, j0[0].transfers_);
  EXPECT_EQ(5, j0[0].duration_);

  /*** CANCEL STOP E IN T1: A@09:00 -> E@10:00 direct via TX ***/
  auto const stats0 = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      mn::to_feed_msg({{.trip_id_ = "T1",
                        .stop_updates_ = {{.stop_id_ = "E", .skip_ = true}}}},
                      date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats0.total_entities_success_);
  auto const r1 =
      mn::route(tags, tt, &rtt,
                mn::make_routing_msg(
                    "tag_A", "tag_E",
                    mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min)),
                metrics);
  auto const j1 = message_to_journeys(motis_content(RoutingResponse, r1));
  ASSERT_EQ(1U, j1.size());
  EXPECT_EQ(0U, j1[0].transfers_);
  EXPECT_EQ(10, j1[0].duration_);

  /*** CANCEL TRIP TX ***/
  auto const stats = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      mn::to_feed_msg({{.trip_id_ = "TX", .cancelled_ = true},
                       {.trip_id_ = "T1",
                        .stop_updates_ = {{.stop_id_ = "A",
                                           .ev_type_ = n::event_type::kDep,
                                           .delay_minutes_ = 10},
                                          {.stop_id_ = "E",
                                           .ev_type_ = n::event_type::kDep,
                                           .skip_ = true},
                                          {.stop_id_ = "B",
                                           .ev_type_ = n::event_type::kDep,
                                           .delay_minutes_ = 5},
                                          {.stop_id_ = "",
                                           .seq_ = 3,
                                           .ev_type_ = n::event_type::kDep,
                                           .stop_assignment_ = "C1"}}},
                       {.trip_id_ = "T2",
                        .stop_updates_ = {{.stop_id_ = "D",
                                           .ev_type_ = n::event_type::kDep,
                                           .delay_minutes_ = 15}}},
                       {.trip_id_ = "T3",
                        .stop_updates_ = {{.stop_id_ = "B",
                                           .ev_type_ = n::event_type::kDep,
                                           .delay_minutes_ = 10},
                                          {.stop_id_ = "C",
                                           .ev_type_ = n::event_type::kArr,
                                           .delay_minutes_ = 0}}},
                       {.trip_id_ = "T4",
                        .stop_updates_ = {{.stop_id_ = "C",
                                           .ev_type_ = n::event_type::kDep,
                                           .delay_minutes_ = 5},
                                          {.stop_id_ = "E",
                                           .ev_type_ = n::event_type::kArr,
                                           .delay_minutes_ = 0}}}},
                      date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(stats.total_entities_success_, 5U);

  auto const routing_response = mn::route(
      tags, tt, &rtt,
      mn::make_routing_msg("tag_A", "tag_E",
                           mn::to_unix(date::sys_days{2019_y / May / 1} + 8h)),
      metrics);

  auto const journeys =
      message_to_journeys(motis_content(RoutingResponse, routing_response));

  std::stringstream ss;
  for (auto const& j : journeys) {
    print_journey(j, ss, false);
  }
  EXPECT_EQ(expected, ss.str());

  /*** BRING BACK TRIP TX ***/
  auto const stats2 = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      mn::to_feed_msg({{.trip_id_ = "TX", .cancelled_ = false}},
                      date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats2.total_entities_success_);
  auto const r2 =
      mn::route(tags, tt, &rtt,
                mn::make_routing_msg(
                    "tag_A", "tag_E",
                    mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min)),
                metrics);
  auto const j2 = message_to_journeys(motis_content(RoutingResponse, r2));
  ASSERT_EQ(1U, j2.size());
  EXPECT_EQ(0U, j2[0].transfers_);
  EXPECT_EQ(10, j2[0].duration_);

  /*** TRIP 1: BRING BACK STOP E + REMOVE DELAY ***/
  auto const stats3 = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      mn::to_feed_msg(
          {{.trip_id_ = "T1",
            .stop_updates_ =
                {{.stop_id_ = "A", .delay_minutes_ = 0, .skip_ = false},
                 {.stop_id_ = "E", .delay_minutes_ = 0, .skip_ = false}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats3.total_entities_success_);
  auto const r3 =
      mn::route(tags, tt, &rtt,
                mn::make_routing_msg(
                    "tag_A", "tag_E",
                    mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min)),
                metrics);
  auto const j3 = message_to_journeys(motis_content(RoutingResponse, r3));
  ASSERT_EQ(1U, j3.size());
  EXPECT_EQ(0U, j3[0].transfers_);
  EXPECT_EQ(5, j3[0].duration_);
}
