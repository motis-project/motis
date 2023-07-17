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
#include "motis/nigiri/routing.h"
#include "motis/nigiri/tag_lookup.h"

#include "./utils.h"

using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;
namespace mn = motis::nigiri;
namespace mm = motis::module;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00, 1h, 1 transfer
// [TRIP 1: A, B, C] -> WALK [5 min] -> [TRIP 2: D, E]
//
// 10:00 - 10:45, 45min, 2 transfers
// [TRIP 1: A, B] -> [TRIP 3: B, C] -> [TRIP 4: C, E]
//
// TIMETABLE + DELAYS:
// TRIP  |  STOP  |  EV  |  PLANNED  | REAL TIME
// ------+--------+------+-----------+-------------
// 1     |  A     | DEP  |  09:50    | 10:00 (+10)
// 1     |  B     | ARR  |  10:00    | 10:10 (+10)
// 1     |  B     | DEP  |  10:10    | 10:15 (+5)
// 1     |  C     | ARR  |  10:30    | 10:35 (+5)
// 2     |  D     | DEP  |  10:25    | 10:40 (+15)
// 2     |  E     | ARR  |  10:45    | 11:00 (+15)
// 3     |  B     | DEP  |  10:10    | 10:20 (+10)
// 3     |  C     | ARR  |  10:30    | 10:30 (+0)
// 4     |  C     | DEP  |  10:30    | 10:35 (+5)
// 4     |  E     | ARR  |  10:45    | 10:45 (+0)
n::loader::mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{n::loader::gtfs::kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{n::loader::gtfs::kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,7.0,8.0,,
)"}},
       {path{n::loader::gtfs::kCalendarDatesFile},
        std::string{R"(service_id,date,exception_type
S1,20190501,1
)"}},
       {path{n::loader::gtfs::kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
)"}},
       {path{n::loader::gtfs::kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R3,S1,T3,,
R4,S1,T4,,
)"}},
       {path{n::loader::gtfs::kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,09:50:00,09:50:00,A,1,0,0
T1,10:00:00,10:10:00,B,2,0,0
T1,10:30:00,10:30:00,C,3,0,0
T2,10:25:00,10:25:00,D,1,0,0
T2,10:45:00,10:45:00,E,2,0,0
T3,10:10:00,10:10:00,B,1,0,0
T3,10:30:00,10:30:00,C,2,0,0
T4,10:30:00,10:30:00,C,1,0,0
T4,10:45:00,10:45:00,E,2,0,0
)"}},
       {path{n::loader::gtfs::kTransfersFile},
        std::string{R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time
C,D,0,2
)"}}}};
}

constexpr auto const expected =
    R"(Journey: duration=60  transfers=1  accessibility=0                01.05. 07:50 +10 F --> 01.05. 08:45 +15 F (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 07:50 +10 F       enter
 1: tag_B   B............................................... a: 01.05. 08:00 +10 F  d: 01.05. 08:10 +5  F
 2: tag_C   C............................................... a: 01.05. 08:30 +5  F  d: 01.05. 08:30 +5  F  exit
 3: tag_D   D............................................... a: 01.05. 08:32 +5  F  d: 01.05. 08:25 +15 F       enter
 4: tag_E   E............................................... a: 01.05. 08:45 +15 F  d:                     exit

Transports:
 0: 0  -> 2  TRAIN Bus 1                     duration=35, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="35"
 1: 2  -> 3  WALK  type=          id=-1      duration=2   accessibility=0
 2: 3  -> 4  TRAIN Bus 2                     duration=20, provider="Deutsche Bahn", direction="", line="2", clasz=3, duration="20"

Trips:
 0: 0  -> 2  {tag_A  ,      0, 2019-05-01 07:50} -> {  tag_C, 2019-05-01 08:30}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697000/tag_C/1556699400/1  ::memory::/stop_times.txt:2:4
 1: 3  -> 4  {tag_D  ,      0, 2019-05-01 08:25} -> {  tag_E, 2019-05-01 08:45}, line_id=2, id=tag_T2
       #/trip/tag_D/0/1556699100/tag_E/1556700300/2  ::memory::/stop_times.txt:5:6

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
       #/trip/tag_A/0/1556697000/tag_C/1556699400/1  ::memory::/stop_times.txt:2:4
 1: 1  -> 2  {tag_B  ,      0, 2019-05-01 08:10} -> {  tag_C, 2019-05-01 08:30}, line_id=3, id=tag_T3
       #/trip/tag_B/0/1556698200/tag_C/1556699400/3  ::memory::/stop_times.txt:7:8
 2: 2  -> 3  {tag_C  ,      0, 2019-05-01 08:30} -> {  tag_E, 2019-05-01 08:45}, line_id=4, id=tag_T4
       #/trip/tag_C/0/1556699400/tag_E/1556700300/4  ::memory::/stop_times.txt:9:10

Attributes:
)";

}  // namespace

TEST(nigiri, rt_test) {
  auto tt = n::timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  n::loader::register_special_stations(tt);
  n::loader::gtfs::load_timetable({}, n::source_idx_t{0}, test_files(), tt);
  n::loader::finalize(tt);

  auto rtt = n::rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});

  auto const stats = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      mn::to_feed_msg(
          {mn::trip{
               .trip_id_ = "T1",
               .delays_ = {mn::trip::delay{.stop_id_ = "A",
                                           .ev_type_ = nigiri::event_type::kDep,
                                           .delay_minutes_ = 10},
                           mn::trip::delay{.stop_id_ = "B",
                                           .ev_type_ = nigiri::event_type::kDep,
                                           .delay_minutes_ = 5}}},
           mn::trip{
               .trip_id_ = "T2",
               .delays_ = {mn::trip::delay{.stop_id_ = "D",
                                           .ev_type_ = nigiri::event_type::kDep,
                                           .delay_minutes_ = 15}}},
           mn::trip{
               .trip_id_ = "T3",
               .delays_ = {mn::trip::delay{.stop_id_ = "B",
                                           .ev_type_ = nigiri::event_type::kDep,
                                           .delay_minutes_ = 10},
                           mn::trip::delay{.stop_id_ = "C",
                                           .ev_type_ = nigiri::event_type::kArr,
                                           .delay_minutes_ = 0}}},
           mn::trip{
               .trip_id_ = "T4",
               .delays_ = {mn::trip::delay{.stop_id_ = "C",
                                           .ev_type_ = nigiri::event_type::kDep,
                                           .delay_minutes_ = 5},
                           mn::trip::delay{.stop_id_ = "E",
                                           .ev_type_ = nigiri::event_type::kArr,
                                           .delay_minutes_ = 0}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(stats.total_entities_success_, 4U);

  auto tags = mn::tag_lookup{};
  tags.add(n::source_idx_t{0U}, "tag_");

  auto const routing_response = mn::route(
      tags, tt, &rtt,
      mn::make_routing_msg("tag_A", "tag_E",
                           mn::to_unix(date::sys_days{2019_y / May / 1} + 8h)));

  using namespace motis;
  using motis::routing::RoutingResponse;
  auto const journeys =
      message_to_journeys(motis_content(RoutingResponse, routing_response));

  std::stringstream ss;
  for (auto const& j : journeys) {
    print_journey(j, ss, false);
  }
  EXPECT_EQ(expected, ss.str());
}