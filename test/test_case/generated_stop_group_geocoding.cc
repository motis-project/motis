#include "motis/config.h"

#include "../test_case.h"

using motis::config;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
G1,Group 1,,0.0,0.0,,0,
G2,Group 2,,0.0,0.0,,0,
A,Stop A,,48.1,11.5,,0,
B,Stop B,,48.2,11.6,,0,
C,Stop C,,48.3,11.7,,0,
D,Stop D,,48.4,11.8,,0,

# stop_group_elements.txt
stop_group_id,stop_id
G1,A
G2,B

# calendar_dates.txt
service_id,date,exception_type
S1,20200101,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RB,DB,RB,,,3
RT,DB,RT,,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RB,S1,TB,RB,
RT,S1,TT,RT,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TB,10:00:00,10:00:00,A,1,0,0
TB,10:30:00,10:30:00,C,2,0,0
TT,11:00:00,11:00:00,B,1,0,0
TT,11:30:00,11:30:00,D,2,0,0
)";

template <>
test_case_params const
import_test_case<test_case::generated_stop_group_geocoding>() {
  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2020-01-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .geocoding_ = true};
  return import_test_case(std::move(c),
                          "test/test_case/generated_stop_group_geocoding_data");
}
