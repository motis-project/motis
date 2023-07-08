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
A,A,,49.878022,8.654508,,
B,B,,49.876101,8.650253,,
C,C,,49.872589,8.631291,,
)"}},
       {path{n::loader::gtfs::kCalendarDatesFile},
        std::string{R"(service_id,date,exception_type
S1,20190501,1
S1,20190502,1
S1,20190503,1
)"}},
       {path{n::loader::gtfs::kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
)"}},
       {path{n::loader::gtfs::kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R1,S1,T2,,
)"}},
       {path{n::loader::gtfs::kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:00:00,00:00:00,A,1,0,0
T1,01:00:00,02:00:00,B,2,0,0
T1,03:00:00,03:00:00,C,3,0,0
T2,00:30:00,00:30:00,A,1,0,0
T2,01:30:00,01:30:00,B,2,0,0
)"}},
       {path{n::loader::gtfs::kFrequenciesFile},
        std::string{R"(trip_id,start_time,end_time,headway_secs,exact_times
T1,00:00:00,23:00:00,3600,1
T2,00:30:00,23:30:00,7200,1
)"}}}};
}

}  // namespace

TEST(nigiri, railviz_test) {
  auto tt = n::timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  n::loader::register_special_stations(tt);
  n::loader::gtfs::load_timetable({}, n::source_idx_t{0}, test_files(), tt);
  n::loader::finalize(tt);

  auto tags = mn::tag_lookup{};
  tags.add(n::source_idx_t{0U}, "tag_");
}