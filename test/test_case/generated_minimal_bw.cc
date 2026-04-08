#include "motis/config.h"

#include "../test_case.h"

using motis::config;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
TEST,Test Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
STOP1,Stop 1,48.0,9.0,0,,
STOP2,Stop 2,48.1,9.1,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,TEST,R1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,STOP1,1,0,0
T1,10:10:00,10:10:00,STOP2,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20260110,1
)";

template <>
test_case_params const import_test_case<test_case::generated_minimal_bw>() {
  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2026-01-10",
                                   .num_days_ = 1,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = false};
  return import_test_case(std::move(c),
                          "test/test_case/generated_minimal_bw_data");
}
