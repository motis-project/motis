#include "motis/endpoints/stop_times.h"
#include "gtest/gtest.h"

#include <chrono>
#include <sstream>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/json.hpp"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"

#include "../util.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;
namespace n = nigiri;

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

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,block_1
U4,S1,U4,,block_1
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_10,2,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, stop_times) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  auto d = import(c, "test/data", true);
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto const stats =
      n::rt::gtfsrt_update_msg(
          *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
          to_feed_msg({trip_update{
                           .trip_ = {.trip_id_ = "ICE",
                                     .start_time_ = {"00:35:00"},
                                     .date_ = {"20190501"}},
                           .stop_updates_ = {{.stop_id_ = "FFM_12",
                                              .seq_ = std::optional{1U},
                                              .ev_type_ = n::event_type::kArr,
                                              .delay_minutes_ = 10,
                                              .stop_assignment_ = "FFM_12"}}},
                       alert{
                           .header_ = "Yeah",
                           .description_ = "Yeah!!",
                           .entities_ = {{.trip_ =
                                              {
                                                  {.trip_id_ = "ICE",
                                                   .start_time_ = {"00:35:00"},
                                                   .date_ = {"20190501"}},
                                              },
                                          .stop_id_ = "DA"}}},
                       alert{.header_ = "Hello",
                             .description_ = "World",
                             .entities_ =
                                 {{.trip_ = {{.trip_id_ = "ICE",
                                              .start_time_ = {"00:35:00"},
                                              .date_ = {"20190501"}}}}}}},
                      date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats.total_entities_success_);
  EXPECT_EQ(2U, stats.alert_total_resolve_success_);

  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  EXPECT_EQ(d.rt_->rtt_.get(), stop_times.rt_->rtt_.get());

  {
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true");

    auto const format_time = [&](auto&& t, char const* fmt = "%F %H:%M") {
      return date::format(fmt, *t);
    };

    EXPECT_EQ("test_FFM_10", res.place_.stopId_);
    EXPECT_EQ(3, res.stopTimes_.size());

    auto const& ice = res.stopTimes_[0];
    EXPECT_EQ(api::ModeEnum::HIGHSPEED_RAIL, ice.mode_);
    EXPECT_EQ("20190501_00:35_test_ICE", ice.tripId_);
    EXPECT_EQ("ICE", ice.displayName_);
    EXPECT_EQ("FFM Hbf", ice.headsign_);
    EXPECT_EQ("ICE", ice.routeId_);
    EXPECT_EQ("2019-04-30 22:55", format_time(ice.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 22:45",
              format_time(ice.place_.scheduledArrival_.value()));
    EXPECT_EQ(true, ice.realTime_);
    EXPECT_EQ(1, ice.previousStops_->size());
    EXPECT_EQ(1, ice.place_.alerts_->size());

    auto const& sbahn = res.stopTimes_[2];
    EXPECT_EQ(
        api::ModeEnum::SUBWAY,
        sbahn.mode_);  // mode can't change with block_id so sticks from U4
    EXPECT_EQ("20190501_01:15_test_S3", sbahn.tripId_);
    EXPECT_EQ("S3", sbahn.displayName_);
    EXPECT_EQ("FFM Hbf", sbahn.headsign_);
    EXPECT_EQ("S3", sbahn.routeId_);
    EXPECT_EQ("2019-04-30 23:20", format_time(sbahn.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 23:20",
              format_time(sbahn.place_.scheduledArrival_.value()));
    EXPECT_EQ(false, sbahn.realTime_);
    EXPECT_EQ(2, sbahn.previousStops_->size());
  }
}
