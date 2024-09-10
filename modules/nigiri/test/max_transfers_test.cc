#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
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

// The schedule contains 3 connections from A to D:
//
// - Direct connection - 10:00 -> 11:00:
//   T1: A (10:00) -> D (11:00)
//
// - One transfer - 10:00 -> 10:50:
//   T2: A (10:00) -> B (10:10)
//   T3: B (10:15) -> D (10:50)
//
// - Two transfers - 10:00 -> 10:40:
//   T2: A (10:00) -> B (10:10)
//   T4: B (10:13) -> C (10:20)
//   T5: C (10:25) -> D (10:40)
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
R5,DB,5,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,TX,,
R1,S1,T1,,
R2,S1,T2,,
R3,S1,T3,,
R4,S1,T4,,
R5,S1,T5,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,0,0,0
T1,11:00:00,11:00:00,D,1,0,0
T2,10:00:00,10:00:00,A,0,0,0
T2,10:10:00,10:10:00,B,1,0,0
T3,10:15:00,10:15:00,B,0,0,0
T3,10:50:00,10:50:00,D,1,0,0
T4,10:13:00,10:13:00,B,0,0,0
T4,10:20:00,10:20:00,C,1,0,0
T5,10:25:00,10:25:00,C,0,0,0
T5,10:40:00,10:40:00,D,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected_max_transfers_default =
    R"(Journey: duration=60  transfers=0  accessibility=0                01.05. 08:00       --> 01.05. 09:00       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_D   D............................................... a: 01.05. 09:00        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 1                     duration=60, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="60"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_D, 2019-05-01 09:00}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697600/tag_D/1556701200/1  ::memory::/stop_times.txt:2:3

Attributes:
Journey: duration=50  transfers=1  accessibility=0                01.05. 08:00       --> 01.05. 08:50       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_B   B............................................... a: 01.05. 08:10        d: 01.05. 08:15        exit enter
 2: tag_D   D............................................... a: 01.05. 08:50        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 2                     duration=10, provider="Deutsche Bahn", direction="", line="2", clasz=3, duration="10"
 1: 1  -> 2  TRAIN Bus 3                     duration=35, provider="Deutsche Bahn", direction="", line="3", clasz=3, duration="35"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_B, 2019-05-01 08:10}, line_id=2, id=tag_T2
       #/trip/tag_A/0/1556697600/tag_B/1556698200/2  ::memory::/stop_times.txt:4:5
 1: 1  -> 2  {tag_B  ,      0, 2019-05-01 08:15} -> {  tag_D, 2019-05-01 08:50}, line_id=3, id=tag_T3
       #/trip/tag_B/0/1556698500/tag_D/1556700600/3  ::memory::/stop_times.txt:6:7

Attributes:
Journey: duration=40  transfers=2  accessibility=0                01.05. 08:00       --> 01.05. 08:40       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_B   B............................................... a: 01.05. 08:10        d: 01.05. 08:13        exit enter
 2: tag_C   C............................................... a: 01.05. 08:20        d: 01.05. 08:25        exit enter
 3: tag_D   D............................................... a: 01.05. 08:40        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 2                     duration=10, provider="Deutsche Bahn", direction="", line="2", clasz=3, duration="10"
 1: 1  -> 2  TRAIN Bus 4                     duration=7, provider="Deutsche Bahn", direction="", line="4", clasz=3, duration="7"
 2: 2  -> 3  TRAIN Bus 5                     duration=15, provider="Deutsche Bahn", direction="", line="5", clasz=3, duration="15"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_B, 2019-05-01 08:10}, line_id=2, id=tag_T2
       #/trip/tag_A/0/1556697600/tag_B/1556698200/2  ::memory::/stop_times.txt:4:5
 1: 1  -> 2  {tag_B  ,      0, 2019-05-01 08:13} -> {  tag_C, 2019-05-01 08:20}, line_id=4, id=tag_T4
       #/trip/tag_B/0/1556698380/tag_C/1556698800/4  ::memory::/stop_times.txt:8:9
 2: 2  -> 3  {tag_C  ,      0, 2019-05-01 08:25} -> {  tag_D, 2019-05-01 08:40}, line_id=5, id=tag_T5
       #/trip/tag_C/0/1556699100/tag_D/1556700000/5  ::memory::/stop_times.txt:10:11

Attributes:
)";

constexpr auto const expected_max_transfers_1 =
    R"(Journey: duration=60  transfers=0  accessibility=0                01.05. 08:00       --> 01.05. 09:00       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_D   D............................................... a: 01.05. 09:00        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 1                     duration=60, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="60"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_D, 2019-05-01 09:00}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697600/tag_D/1556701200/1  ::memory::/stop_times.txt:2:3

Attributes:
Journey: duration=50  transfers=1  accessibility=0                01.05. 08:00       --> 01.05. 08:50       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_B   B............................................... a: 01.05. 08:10        d: 01.05. 08:15        exit enter
 2: tag_D   D............................................... a: 01.05. 08:50        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 2                     duration=10, provider="Deutsche Bahn", direction="", line="2", clasz=3, duration="10"
 1: 1  -> 2  TRAIN Bus 3                     duration=35, provider="Deutsche Bahn", direction="", line="3", clasz=3, duration="35"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_B, 2019-05-01 08:10}, line_id=2, id=tag_T2
       #/trip/tag_A/0/1556697600/tag_B/1556698200/2  ::memory::/stop_times.txt:4:5
 1: 1  -> 2  {tag_B  ,      0, 2019-05-01 08:15} -> {  tag_D, 2019-05-01 08:50}, line_id=3, id=tag_T3
       #/trip/tag_B/0/1556698500/tag_D/1556700600/3  ::memory::/stop_times.txt:6:7

Attributes:
)";

constexpr auto const expected_max_transfers_0 =
    R"(Journey: duration=60  transfers=0  accessibility=0                01.05. 08:00       --> 01.05. 09:00       (UTC)

Stops:
 0: tag_A   A............................................... a:                     d: 01.05. 08:00             enter
 1: tag_D   D............................................... a: 01.05. 09:00        d:                     exit

Transports:
 0: 0  -> 1  TRAIN Bus 1                     duration=60, provider="Deutsche Bahn", direction="", line="1", clasz=3, duration="60"

Trips:
 0: 0  -> 1  {tag_A  ,      0, 2019-05-01 08:00} -> {  tag_D, 2019-05-01 09:00}, line_id=1, id=tag_T1
       #/trip/tag_A/0/1556697600/tag_D/1556701200/1  ::memory::/stop_times.txt:2:3

Attributes:
)";

}  // namespace

TEST(nigiri, max_transfers_test) {
  using namespace motis;
  using motis::routing::RoutingResponse;

  auto tt = n::timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  n::loader::register_special_stations(tt);
  n::loader::gtfs::load_timetable({}, n::source_idx_t{0},
                                  n::loader::mem_dir::read(test_files), tt);
  n::loader::finalize(tt);

  auto tags = mn::tag_lookup{};
  tags.add(n::source_idx_t{0U}, "tag_");

  auto prometheus_registry = prometheus::Registry{};
  auto metrics = mn::metrics{prometheus_registry};

  {  // Default transfer limit -> should find all 3 connections
    auto const results = mn::route(
        tags, tt, nullptr,
        mn::make_routing_msg(
            "tag_A", "tag_D",
            mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min)),
        metrics);

    auto const journeys =
        message_to_journeys(motis_content(RoutingResponse, results));

    std::stringstream ss;
    for (auto const& j : journeys) {
      print_journey(j, ss, false);
    }
    EXPECT_EQ(expected_max_transfers_default, ss.str());
  }

  {  // Max 1 transfer -> should find 2 connections
    auto const results = mn::route(
        tags, tt, nullptr,
        mn::make_routing_msg(
            "tag_A", "tag_D",
            mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min), 1),
        metrics);

    auto const journeys =
        message_to_journeys(motis_content(RoutingResponse, results));

    std::stringstream ss;
    for (auto const& j : journeys) {
      print_journey(j, ss, false);
    }
    EXPECT_EQ(expected_max_transfers_1, ss.str());
  }

  {  // Max 0 transfers -> should find only the direct connection
    auto const results = mn::route(
        tags, tt, nullptr,
        mn::make_routing_msg(
            "tag_A", "tag_D",
            mn::to_unix(date::sys_days{2019_y / May / 1} + 7h + 50min), 0),
        metrics);

    auto const journeys =
        message_to_journeys(motis_content(RoutingResponse, results));

    std::stringstream ss;
    for (auto const& j : journeys) {
      print_journey(j, ss, false);
    }
    EXPECT_EQ(expected_max_transfers_0, ss.str());
  }
}
