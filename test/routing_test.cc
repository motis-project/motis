#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "osr/extract/extract.h"
#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "icc/compute_footpaths.h"
#include "icc/elevators/elevators.h"
#include "icc/elevators/match_elevator.h"
#include "icc/endpoints/routing.h"
#include "icc/match_platforms.h"
#include "icc/tt_location_rtree.h"

namespace n = nigiri;
namespace nl = nigiri::loader;
namespace json = boost::json;
using namespace std::string_view_literals;
using namespace icc;
using namespace date;

constexpr auto const kFastaJson = R"__(
[
  {
    "description": "FFM HBF zu Gleis 101/102 (S-Bahn)",
    "equipmentnumber" : 10561326, "geocoordX" : 8.6628995,
    "geocoordY" : 50.1072933, "operatorname" : "DB InfraGO",
    "state" : "ACTIVE",
    "stateExplanation" : "available",
    "stationnumber" : 1866,
    "type" : "ELEVATOR"
  },
  {
    "description": "FFM HBF zu Gleis 103/104 (S-Bahn)",
    "equipmentnumber": 10561327,
    "geocoordX": 8.6627516,
    "geocoordY": 50.1074549,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1866,
    "type": "ELEVATOR"
  },
  {
    "description": "HAUPTWACHE zu Gleis 2/3 (S-Bahn)",
    "equipmentnumber": 10351032,
    "geocoordX": 8.67818,
    "geocoordY": 50.114046,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1864,
    "type": "ELEVATOR"
  },
  {
    "description": "DA HBF zu Gleis 1",
    "equipmentnumber": 10543458,
    "geocoordX": 8.6303864,
    "geocoordY": 49.8725612,
    "state": "ACTIVE",
    "type": "ELEVATOR"
  },
  {
    "description": "DA HBF zu Gleis 3/4",
    "equipmentnumber": 10543453,
    "geocoordX": 8.6300911,
    "geocoordY": 49.8725678,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 5/6",
    "equipmentnumber": 10543454,
    "geocoordX": 8.6298163,
    "geocoordY": 49.8725555,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 7/8",
    "equipmentnumber": 10543455,
    "geocoordX": 8.6295535,
    "geocoordY": 49.87254,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 9/10",
    "equipmentnumber": 10543456,
    "geocoordX": 8.6293117,
    "geocoordY": 49.8725263,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 11/12",
    "equipmentnumber": 10543457,
    "geocoordX": 8.6290451,
    "geocoordY": 49.8725147,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  }
]
)__"sv;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA_HBF,3
DA_10,DA Hbf,49.87336,8.62926,0,DA_HBF,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10773,8.66322,0,FFM_HBF,101
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM_HBF,12
FFM_U,FFM Hbf,50.107577,8.6638173,0,FFM_HBF,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
RB,DB,RB,,,106
U4,DB,U4,,,402

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
RB,S1,RB,,
U4,S1,U4,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,00:30:00,00:30:00,DA_3,0,0,0
S3,00:50:00,00:50:00,LANGEN,1,0,0
S3,01:20:00,00:20:00,FFM_HAUPT_S,2,0,0
S3,01:25:00,01:25:00,FFM_101,3,0,0
RB,00:35:00,00:35:00,DA_10,0,0,0
RB,00:45:00,00:45:00,LANGEN,1,0,0
RB,00:55:00,00:55:00,FFM_12,2,0,0
U4,01:05:00,01:10:00,FFM_U,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

TEST(a, b) {
  constexpr auto const kOsrPath = "test/test_case_osr";

  // Load OSR.
  osr::extract(true, "test/resources/test_case.osm.pbf", kOsrPath);
  auto const w = osr::ways{kOsrPath, cista::mmap::protection::READ};
  auto pl = osr::platforms{kOsrPath, cista::mmap::protection::READ};
  auto const l = osr::lookup{w};
  auto const elevator_nodes = get_elevator_nodes(w);
  auto const e =
      std::make_shared<elevators>(w, elevator_nodes, parse_fasta(kFastaJson));
  pl.build_rtree(w);

  // Load timetable.
  auto tt = n::timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  nl::register_special_stations(tt);
  nl::gtfs::load_timetable({}, n::source_idx_t{}, nl::mem_dir::read(kGTFS), tt);
  nl::finalize(tt);
  auto const loc_rtree = create_location_rtree(tt);

  // Compute footpaths.
  compute_footpaths(tt, w, l, pl, e->blocked_, true);

  // Init real-time timetable.
  auto const today = date::sys_days{2019_y / May / 1};
  auto rtt =
      std::make_shared<n::rt_timetable>(n::rt::create_rt_timetable(tt, today));

  // Match platforms.
  auto const matches = get_matches(tt, pl, w);

  // Instantiate routing endpoint.
  auto const routing = ep::routing{w, l, pl, tt, rtt, e, loc_rtree, matches};
  auto const plan_response =
      routing("/?fromPlace=DA&toPlace=FFM_HAUPT&date=04-30-2019&time=22:00");

  std::cout << json::serialize(json::value_from(plan_response)) << "\n";
}