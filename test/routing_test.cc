#include "gtest/gtest.h"

#include "boost/json.hpp"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

namespace json = boost::json;
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
    out << "from=" << leg.from_.stopId_.value_or("-")
        << ", to=" << leg.to_.stopId_.value_or("-") << ", ";
    out << "start=";
    format_time(leg.startTime_);
    out << ", mode=";
    out << json::serialize(json::value_from(leg.mode_));
    out << ", end=";
    format_time(leg.endTime_);
    out << ")";
  }
  out << "\n]";
}

TEST(motis, routing) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto d = import(
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{
                     .first_day_ = "2019-05-01",
                     .num_days_ = 2,
                     .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}},
             .street_routing_ = true,
             .osr_footpath_ = true},
      "test/data", false);
  d.rt_->e_ = std::make_unique<elevators>(*d.w_, *d.elevator_nodes_,
                                          parse_fasta(kFastaJson));
  auto const routing = utl::init_from<ep::routing>(d).value();

  // Route with wheelchair.
  {
    auto const plan_response = routing(
        "/?fromPlace=49.87263,8.63127&toPlace=50.11347,8.67664"
        "&date=05-01-2019&time=01:25&wheelchair=true");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.itineraries_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:29, end=02:29, duration=01:04, transfers=1, legs=[
    (from=-, to=test_DA_10, start=2019-05-01 01:29, mode="WALK", end=2019-05-01 01:35),
    (from=test_DA_10, to=test_FFM_12, start=2019-05-01 01:35, mode="HIGHSPEED_RAIL", end=2019-05-01 01:45),
    (from=test_FFM_12, to=test_FFM_101, start=2019-05-01 01:45, mode="WALK", end=2019-05-01 01:51),
    (from=test_FFM_101, to=test_FFM_HAUPT_S, start=2019-05-01 02:15, mode="METRO", end=2019-05-01 02:20),
    (from=test_FFM_HAUPT_S, to=-, start=2019-05-01 02:20, mode="WALK", end=2019-05-01 02:29)
])",
        ss.str());
  }

  // Route without wheelchair.
  {
    auto const plan_response = routing(
        "/?fromPlace=49.87263,8.63127&toPlace=50.11347,8.67664"
        "&date=05-01-2019&time=01:25");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.itineraries_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:25, end=02:14, duration=00:49, transfers=1, legs=[
    (from=-, to=test_DA_10, start=2019-05-01 01:25, mode="WALK", end=2019-05-01 01:28),
    (from=test_DA_10, to=test_FFM_12, start=2019-05-01 01:35, mode="HIGHSPEED_RAIL", end=2019-05-01 01:45),
    (from=test_FFM_12, to=test_de:6412:10:6:1, start=2019-05-01 01:45, mode="WALK", end=2019-05-01 01:49),
    (from=test_de:6412:10:6:1, to=test_FFM_HAUPT_U, start=2019-05-01 02:05, mode="SUBWAY", end=2019-05-01 02:10),
    (from=test_FFM_HAUPT_U, to=-, start=2019-05-01 02:10, mode="WALK", end=2019-05-01 02:14)
])",
        ss.str());
  }
}