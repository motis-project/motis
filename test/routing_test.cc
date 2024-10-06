#include "gtest/gtest.h"

#include <map>

#include "boost/json.hpp"

#include "utl/init_from.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "osr/extract/extract.h"
#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "motis/compute_footpaths.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/match_elevator.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/tt_location_rtree.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
namespace nl = nigiri::loader;
namespace json = boost::json;
namespace fs = std::filesystem;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;

constexpr auto const kFastaJson = R"__(
[
  {
    "description": "FFM HBF zu Gleis 101/102 (S-Bahn)",
    "equipmentnumber" : 10561326,
    "geocoordX" : 8.6628995,
    "geocoordY" : 50.1072933,
    "operatorname" : "DB InfraGO",
    "state" : "ACTIVE",
    "stateExplanation" : "available",
    "stationnumber" : 1866,
    "type" : "ELEVATOR",
    "outOfService": [
      ["2019-05-01T01:30:00Z", "2019-05-01T02:30:00Z"]
    ]
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
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
RB,DB,RB,,,106
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
RB,S1,RB,,
U4,S1,U4,,
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
RB,00:35:00,00:35:00,DA_10,0,0,0
RB,00:45:00,00:45:00,LANGEN,1,0,0
RB,00:55:00,00:55:00,FFM_12,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:45:00,00:45:00,DA_10,0,0,0
ICE,00:55:00,00:55:00,FFM_12,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
S3,01:15:00,25:15:00,3600
RB,00:35:00,24:35:00,3600
ICE,00:35:00,24:35:00,3600
U4,01:05:00,25:01:00,3600
)"sv;

void print_short(std::ostream& out, api::Itinerary const& j) {
  auto const format_time = [&](auto&& t, char const* fmt = "%F %H:%M") {
    auto const u = std::chrono::time_point<std::chrono::system_clock>{
        std::chrono::milliseconds{t}};
    out << date::format(fmt, u);
  };

  out << "date=";
  format_time(j.startTime_, "%F");
  out << ", start=";
  format_time(j.startTime_, "%H:%M");
  out << ", end=";
  format_time(j.endTime_, "%H:%M");

  out << ", duration=";
  format_time(j.duration_ * 1000U, "%H:%M");
  out << ", transfers=" << j.transfers_;

  out << ", legs=[\n";
  auto first = true;
  for (auto const& leg : j.legs_) {
    if (!first) {
      out << ",\n    ";
    } else {
      out << "    ";
    }
    first = false;
    out << "(";
    out << "start=";
    format_time(leg.startTime_);
    out << ", mode=";
    out << json::serialize(json::value_from(leg.mode_));
    out << ", end=";
    format_time(leg.endTime_);
    out << ")";
  }
  out << "\n]\n";
}

void load(fs::path const& data_path,
          n::interval<date::sys_days> const interval,
          std::string_view gtfs) {
  auto ec = std::error_code{};
  fs::remove_all(data_path, ec);
  fs::create_directories(data_path, ec);

  // Load OSR.
  auto const osr_path = data_path / "osr";
  osr::extract(true, "test/resources/test_case.osm.pbf", osr_path);
  auto const w = osr::ways{osr_path, cista::mmap::protection::READ};
  auto pl = osr::platforms{osr_path, cista::mmap::protection::READ};
  auto const l = osr::lookup{w};
  auto const elevator_nodes = get_elevator_nodes(w);
  pl.build_rtree(w);

  // Load assistance times.
  auto assistance = n::loader::read_assistance(R"(name,lat,lng,time
DA HBF,49.87260,8.63085,06:15-22:30
FFM,50.10701,8.66341,06:15-22:30
)");

  // Load timetable.
  auto tt = n::timetable{};
  tt.date_range_ = interval;
  nl::register_special_stations(tt);
  nl::gtfs::load_timetable({}, n::source_idx_t{}, nl::mem_dir::read(gtfs), tt,
                           &assistance);
  nl::finalize(tt);

  fmt::println("computing footpaths");
  auto const elevator_footpath_map = compute_footpaths(w, l, pl, tt, true);

  fmt::println("writing elevator footpaths");
  write(data_path / "elevator_footpath_map.bin", elevator_footpath_map);

  fmt::println("writing timetable");
  tt.write(data_path / "tt.bin");

  std::ofstream{data_path / "fasta.json"}.write(kFastaJson.data(),
                                                kFastaJson.size());
}

TEST(motis, routing) {
  auto const data_path = fs::path{"test/data"};

  load(data_path,
       {date::sys_days{2019_y / March / 25},
        date::sys_days{2019_y / November / 1}},
       kGTFS);

  auto d = data{
      data_path,
      config{.features_ = {{feature::OSR_FOOTPATH, feature::ELEVATORS,
                            feature::TIMETABLE, feature::STREET_ROUTING}}}};
  auto const routing = utl::init_from<ep::routing>(d).value();

  // Route with wheelchair.
  {
    auto const plan_response = routing(
        "/?fromPlace=49.87263,8.63127&toPlace=50.11347,8.67664"
        "&date=05-01-2019&time=01:25&wheelchair=true");

    std::cout << "With wheelchair:\n";
    for (auto const& j : plan_response.itineraries_) {
      print_short(std::cout, j);
      std::cout << "\n";
    }
  }

  // Route without wheelchair.
  {
    auto const plan_response = routing(
        "/?fromPlace=49.87263,8.63127&toPlace=50.11347,8.67664"
        "&date=05-01-2019&time=01:25");

    std::cout << "Without wheelchair:\n";
    for (auto const& j : plan_response.itineraries_) {
      print_short(std::cout, j);
      std::cout << "\n";
    }
  }
}