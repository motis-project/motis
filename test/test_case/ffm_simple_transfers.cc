#include "motis/config.h"

#include "../test_case.h"

using motis::config;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
DA_Bus_1,DA Hbf,49.8724891,8.6281994
DA_Bus_2,DA Hbf,49.8750407,8.6312172
DA_Tram_1,DA Hbf,49.8742551,8.6321063
DA_Tram_2,DA Hbf,49.8731133,8.6313674
DA_Tram_3,DA Hbf,49.872435,8.632164

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
B1,Test,B1,,3
T1,Test,T1,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign
B1,S1,B1,Bus 1,
T1,S1,T1,Tram 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
B1,01:00:00,01:00:00,DA_Bus_1,1
B1,01:10:00,01:10:00,DA_Bus_2,2
T1,01:05:00,01:05:00,DA_Tram_1,1
T1,01:15:00,01:15:00,DA_Tram_2,2
T1,01:20:00,01:20:00,DA_Tram_3,3

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

template <>
data import_test_case<test_case::FFM_simple_transfers>() {
  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .with_shapes_ = true,
              .datasets_ = {{"test", {.path_ = kGTFS}}},
              .route_shapes_ = {{.mode_ =
                                     config::timetable::route_shapes::mode::all,
                                 .cache_db_size_ = 1024U * 1024U * 5U}}},
      .street_routing_ = true};
  return import_test_case(c, "test/test_case/ffm_simple_transfers_data");
}
