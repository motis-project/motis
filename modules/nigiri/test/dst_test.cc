#include "gtest/gtest.h"

#include <string_view>

#include "utl/parser/cstr.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/nigiri/metrics.h"
#include "motis/nigiri/resolve_run.h"
#include "motis/nigiri/routing.h"
#include "motis/nigiri/tag_lookup.h"
#include "motis/nigiri/unixtime_conv.h"

#include "./utils.h"

using namespace date;
namespace n = nigiri;
namespace mn = motis::nigiri;
using namespace std::string_view_literals;
using namespace std::chrono_literals;

constexpr auto const gtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0848 44 66 88"

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
"8101236","Feldkirch","47.2409786035422","9.6041042995269","","Parent8101236"
"8509404:0:4","Buchs SG","47.1684141304207","9.47863660377646","","Parent8509404"
"8503000:0:9","Zürich HB","47.3782796148195","8.53824423161787","","Parent8503000"

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color
"91-1V-Y-j23-1","11","RJX","","RJX","102",,aB0020,FfFfFf

# trips.txt
route_id,service_id,trip_id,trip_headsign,trip_short_name,direction_id,block_id
"91-1V-Y-j23-1","TA+xce80","112.TA.91-1V-Y-j23-1.21.R","Zürich HB","162","1",""

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
"112.TA.91-1V-Y-j23-1.21.R","15:48:00","15:48:00","8101236","1","0","0"
"112.TA.91-1V-Y-j23-1.21.R","16:05:00","16:11:00","8509404:0:4","2","0","0"
"112.TA.91-1V-Y-j23-1.21.R","17:20:00","17:20:00","8503000:0:9","3","0","0"

# calendar.txt
"TA+xce80","1","1","1","1","1","0","0","20221211","20231209"

# calendar_dates.txt
service_id,date,exception_type
"TA+xce80","20231028","1"
"TA+xce80","20231029","1"
"TA+xce80","20231030","2"
"TA+xce80","20231031","2"
"TA+xce80","20231101","2"
)"sv;

TEST(nigiri, dst_test) {
  auto tt = n::timetable{};
  tt.date_range_ = {date::sys_days{2023_y / October / 28},
                    date::sys_days{2023_y / November / 1}};
  n::loader::register_special_stations(tt);
  n::loader::gtfs::load_timetable(
      {.link_stop_distance_ = 0U, .default_tz_ = "Europe/Berlin"},
      n::source_idx_t{0}, n::loader::mem_dir::mem_dir::read(gtfs), tt);
  n::loader::finalize(tt);

  auto tags = mn::tag_lookup{};
  tags.add(n::source_idx_t{0U}, "swiss_");

  auto prometheus_registry = prometheus::Registry{};
  auto metrics = mn::metrics{prometheus_registry};
  auto const routing_response = mn::route(
      tags, tt, nullptr,
      mn::make_routing_msg(
          "swiss_8101236", "swiss_8503000:0:9",
          mn::to_unix(date::sys_days{2023_y / October / 29} + 14h + 48min)),
      metrics);

  using namespace motis;
  using motis::routing::RoutingResponse;
  auto const res = motis_content(RoutingResponse, routing_response);

  ASSERT_EQ(1U, res->connections()->size());
  ASSERT_EQ(1U, res->connections()->Get(0)->trips()->size());
  auto const et =
      to_extern_trip(res->connections()->Get(0)->trips()->Get(0)->id());
  auto const run = mn::resolve_run(tags, tt, et);
  EXPECT_TRUE(run.valid());

  auto const move = res->connections()->Get(0)->transports()->Get(0);
  ASSERT_EQ(move->move_type(), Move_Transport);
  auto const transport = static_cast<const Transport*>(move->move());
  EXPECT_TRUE(transport->route_color());
  ASSERT_EQ(transport->route_color()->str(), "ab0020");
  ASSERT_EQ(transport->route_text_color()->str(), "ffffff");
}
