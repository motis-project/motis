#include "motis/config.h"

#include "../test_case.h"

using motis::config;

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
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,
PAUL1,Römer/Paulskirche,50.110979,8.682276,0,,
PAUL2,Römer/Paulskirche,50.110828,8.681587,0,,
FFM_C,FFM C,50.107812,8.664628,0,,
FFM_B,FFM B,50.107519,8.664775,0,,
DA_Bus_1,DA Hbf,49.8724891,8.6281994
DA_Bus_2,DA Hbf,49.8755778,8.6240518
DA_Tram_1,DA Hbf,49.875345,8.6279307
DA_Tram_2,DA Hbf,49.874995,8.6313925
DA_Tram_3,DA Hbf,49.871561,8.6320181

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101
11_1,DB,11,,,0
11_2,DB,11,,,0
B1,DB,B1,,3
T1,DB,T1,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,block_1
U4,S1,U4,,block_1
ICE,S1,ICE,,
11_1,S1,11_1_1,,
11_1,S1,11_1_2,,
11_2,S1,11_2_1,,
11_2,S1,11_2_2,,
B1,S1,B1,Bus 1,
T1,S1,T1,Tram 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0
11_1_1,12:00:00,12:00:00,PAUL1,0,0,0
11_1_1,12:10:00,12:10:00,FFM_C,1,0,0
11_1_2,12:15:00,12:15:00,PAUL1,0,0,0
11_1_2,12:25:00,12:25:00,FFM_C,1,0,0
11_2_1,12:05:00,12:05:00,FFM_B,0,0,0
11_2_1,12:15:00,12:15:00,PAUL2,1,0,0
11_2_2,12:20:00,12:20:00,FFM_B,0,0,0
11_2_2,12:30:00,12:30:00,PAUL2,1,0,0
B1,00:10:00,00:10:00,DA_Bus_1,1
B1,00:20:00,00:20:00,DA_Bus_2,2
T1,00:24:00,00:24:00,DA_Tram_1,1
T1,00:25:00,00:25:00,DA_Tram_2,2
T1,00:26:00,00:26:00,DA_Tram_3,3

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

template <>
test_case_params const import_test_case<test_case::FFM_for_first_last_mile>() {
  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true};
  return import_test_case(std::move(c),
                          "test/test_case/ffm_for_first_last_mile_data");
}
