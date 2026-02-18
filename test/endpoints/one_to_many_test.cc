#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/stop_times.h"  //  TODO Delete

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

#include "nigiri/common/parse_time.h"
#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/one_to_many.h"
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
PAUL1,Römer/Paulskirche,50.110979,8.682276,0,,
PAUL2,Römer/Paulskirche,50.110828,8.681587,0,,
FFM_C,FFM C,50.107812,8.664628,0,,
FFM_B,FFM B,50.107519,8.664775,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101
11_1,DB,11,,,0
11_2,DB,11,,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,block_1
U4,S1,U4,,block_1
ICE,S1,ICE,,
11_1,S1,11_1_1,,
11_1,S1,11_1_2,,
11_2,S1,11_2_1,,
11_2,S1,11_2_2,,

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

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, one_to_many) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true};
  auto d = import(c, "test/data", true);

  auto const one_to_many_get =
      utl::init_from<ep::one_to_many_intermodal>(d).value();
  auto const one_to_many_post =
      utl::init_from<ep::one_to_many_intermodal_post>(d).value();
  // GET Request, forward
  {
    auto const durations = one_to_many_get(
        "/api/experimental/one-to-many-intermodal?one=49.8722439;8.6320624"
        "&many="
        "49.87336;8.62926,"  // DA_10
        "50.10593;8.66118,"  // FFM_10
        "50.107577;8.6638173,"  // de:6412:10:6:1
        "50.10739;8.66333,"  // FFM_101
        "50.11385;8.67912,"  // FFM_HAUPT_U
        "50.11404;8.67824,"  // FFM_HAUPT_S
        "49.872855;8.632008,"  // Near one
        "49.872504;8.628988,"  // Inside DA station
        "49.874399;8.630361,"  // Still reachable, near DA
        "49.875368;8.627596,"  // Far, near DA
        "50.106596;8.663485,"  // Inside FFM station
        "50.105021;8.663308,"  // Outside FFM station
        "50.107654;8.669103,"  // Far, near FFM
        "50.113494;8.679129,"  // Near FFM_HAUPT_U
        "50.114080;8.677027,"  // Near FFM_HAUPT_S
        "50.114520;8.673050,"  // Too far from FFM_HAUPT_U
        "50.114773;8.672604"  // Far, near FFM_HAUPT
        "&time=2019-04-30T22:30:00.000Z"
        "&maxTravelTime=3600"  // TODO To minutes
        "&maxMatchingDistance=250"
        "&maxDirectTime=540"
        "&maxPostTransitTime=420"
        "&directModes=WALK"
        "&arriveBy=false");

    EXPECT_EQ((std::vector<api::Duration>{{282.0},
                                          {1080.0},
                                          {1380.0},
                                          {1380.0},
                                          {2580.0},
                                          {2580.0},
                                          {123.0},
                                          {242.0},
                                          {530.0},
                                          {},
                                          {1260.0},
                                          {1440.0},
                                          {},
                                          {2700.0},
                                          {2640.0},
                                          {2940.0},
                                          {}}),
              durations);
  }
  // POST Request, backward
  {
    auto const durations = one_to_many_post(api::OneToManyIntermodalParams{
        .one_ = "50.113816,8.679421,0",
        .many_ =
            {
                "49.87336,8.62926",  // DA_10
                "50.10593,8.66118",  // FFM_10
                // "test_FFM_10",  // TODO No result
                "50.107577,8.6638173",  // de:6412:10:6:1
                "50.10739,8.66333",  // FFM_101
                "test_FFM_101",
                "50.11385,8.67912",  // FFM_HAUPT_U
                "test_FFM_HAUPT_U",
                "50.11404,8.67824",  // FFM_HAUPT_S
                "50.113385,8.678328,0",  // Close, near FFM_HAUPT, level 0
                "50.113385,8.678328,-2",  // Close, near FFM_HAUPT, level -2
                "50.111900,8.675208",  // Far, near FFM_HAUPT
                "50.106543,8.663474,0",  // Close, near FFM
                "50.106941,8.659617",  // Too far from de:6412:10:6:1
                "50.104298,8.660285",  // Far, near FFM
                "49.872243,8.632062",  // Near DA
                "49.875368,8.627596",  // Far, near DA
            },
        .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
            n::parse_time("2019-05-01T01:25:00.000+02:00", "%FT%T%Ez"))},
        .maxTravelTime_ = 3600,
        .maxMatchingDistance_ = 250.0,
        .directModes_ = {{api::ModeEnum::WALK}},  // TODO Should be default
        .arriveBy_ = true,
        .maxPreTransitTime_ = 360,
        .maxDirectTime_ = 300});

    EXPECT_EQ((std::vector<api::Duration>{{3180.0},
                                          {1020.0},
                                          // {1020.0},
                                          {780.0},
                                          {780.0},
                                          {720.0},
                                          {159.0},
                                          {159.0},
                                          {127.0},
                                          {103.0},
                                          {123.0},
                                          {},
                                          {900.0},
                                          {1080.0},
                                          {},
                                          {3420.0},
                                          {}}),
              durations);
  }
  // POST, forward, routed, short pre-transit
  {
    auto const durations = one_to_many_post(api::OneToManyIntermodalParams{
        .one_ = "50.106941,8.659617",
        .many_ =
            {
                "test_DA_10",
                "50.107577,8.6638173",  // de:6412:10:6:1
                "test_FFM_HAUPT_S",
                "50.11385,8.67912",  // FFM_HAUPT_U
                "50.105884,8.664241",  // Near FFM
                "50.113291,8.678321,0",  // Near FFM_HAUPT
                "50.113127,8.678879,-2",  // Near FFM_HAUPT
                "50.114141,8.677025,-3",  // Near FFM_HAUPT
                "50.113589,8.679070,-4",  // Near FFM_HAUPT
            },
        .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
            n::parse_time("2019-05-01T00:55:00.000+02:00", "%FT%T%Ez"))},
        .maxTravelTime_ = 3600,
        .maxMatchingDistance_ = 250.0,
        .directModes_ = {{api::ModeEnum::WALK}},  // TODO Should be default
        .arriveBy_ = false,
        .useRoutedTransfers_ = true,
        .maxPreTransitTime_ = 360});  // Too short to reach U4

    EXPECT_EQ((std::vector<api::Duration>{{},
                                          {459.0},
                                          {1620.0},
                                          {1740.0},
                                          {397.0},
                                          {1800.0},
                                          {1800.0},
                                          {1740.0},
                                          {1740.0}}),
              durations);
  }
  // GET, backward, with wheelchair, short post-transit
  {
    auto const durations = one_to_many_get(
        "/api/experimental/one-to-many-intermodal"
        "?one=50.11385;8.67912"  // FFM_HAUPT_U
        "&many="
        "50.107577;8.6638173,"  // de:6412:10:6:1
        "50.10739;8.66333,"  // FFM_101
        "50.11404;8.67824,"  // FFM_HAUPT_S
        "50.113465;8.678477,"  // Near FFM_HAUPT
        "50.112519;8.676565"  // Far, near FFM_HAUPT
        "&time=2019-04-30T23:30:00.000Z"
        "&maxTravelTime=3600"  // TODO To minutes
        "&maxMatchingDistance=250"
        "&maxDirectTime=540"
        "&maxPostTransitTime=240"
        "&directModes=WALK"
        "&pedestrianProfile=WHEELCHAIR"
        "&arriveBy=true");

    EXPECT_EQ((std::vector<api::Duration>{{1680.0},
                                          {},  // Cannot leave from U4
                                          {281.0},
                                          {404.0},
                                          {}}),
              durations);
  }
  // Oneway direction tests
  {
    // GET, forward, preTransitModes + direct
    {
      auto const durations = one_to_many_get(
          "/api/experimental/one-to-many-intermodal"
          "?one=50.107328;8.664836"
          "&many="
          "50.107812;8.664628,"  // FFM C  (shorter path)
          "50.107519;8.664775,"  // FFM B  (longer path, due to oneway)
          "50.110828;8.681587"  // PAUL2
          "&time=2019-05-01T10:00:00.00Z"
          "&maxTravelTime=3600"  // TODO To minutes
          "&maxMatchingDistance=250"
          "&maxDirectTime=3600"
          "&directModes=BIKE"
          "&preTransitModes=BIKE"
          "&arriveBy=false"
          "&cyclingSpeed=2.4");

      EXPECT_EQ((std::vector<api::Duration>{{229.0}, {321.0}, {1980.0}}),
                durations);
    }
    // POST, backward, postTransitModes + direct
    {
      auto const durations = one_to_many_post(api::OneToManyIntermodalParams{
          .one_ = "50.107326,8.665237",
          .many_ = {"test_FFM_B", "test_FFM_C", "test_PAUL1"},
          .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
              n::parse_time("2019-05-01T12:30:00.000+02:00", "%FT%T%Ez"))},
          .maxTravelTime_ = 3600,
          .maxMatchingDistance_ = 250.0,
          .directModes_ = {{api::ModeEnum::BIKE}},
          .postTransitModes_ = {{api::ModeEnum::BIKE}},
          .arriveBy_ = true,
          .cyclingSpeed_ = 2.2,
          .maxDirectTime_ = 1800});

      EXPECT_EQ((std::vector<api::Duration>{{228.0}, {335.0}, {1920.0}}),
                durations);
    }
    // POST, forward, postTransitModes
    {
      auto const durations = one_to_many_post(api::OneToManyIntermodalParams{
          .one_ = "test_PAUL1",
          .many_ = {"test_FFM_C", "50.107326,8.665237"},
          .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
              n::parse_time("2019-05-01T12:00:00.000+02:00", "%FT%T%Ez"))},
          .maxTravelTime_ = 1800,
          .maxMatchingDistance_ = 250.0,
          .postTransitModes_ = {{api::ModeEnum::BIKE}},
          .arriveBy_ = false});

      EXPECT_EQ((std::vector<api::Duration>{{720.0}, {900.0}}), durations);
    }
    // GET, backward, preTransitModes
    {
      auto const durations = one_to_many_get(
          "/api/experimental/one-to-many-intermodal"
          "?one=50.110828;8.681587"  // PAUL2
          "&many="
          "50.107812;8.664628,"  // FFM C
          "50.107519;8.664775,"  // FFM B
          "50.107328;8.664836"  // Long preTransit due to oneway
          "&time=2019-05-01T10:20:00.00Z"
          "&maxTravelTime=3600"  // TODO To minutes
          "&maxMatchingDistance=250"
          "&preTransitModes=BIKE"
          "&arriveBy=true"
          "&cyclingSpeed=2.4");

      EXPECT_EQ((std::vector<api::Duration>{{1140.0}, {1080.0}, {1380.0}}),
                durations);
    }
  }
  // Bug: Should not connect final footpath with first or last mile
  {
    auto const many = std::vector<std::string>{
        "test_FFM_HAUPT_U", "50.11385,8.67912,-4",  // FFM_HAUPT_U
        "test_FFM_HAUPT_S", "50.11404,8.67824,-3",  // FFM_HAUPT_S
        "50.114093,8.676546"};  // Test location
    {
      // Last location should not be reachable when only arriving with U4
      auto const durations = one_to_many_post(api::OneToManyIntermodalParams{
          .one_ = "50.108056,8.663177,-2",  // Near de:6412:10:6:1
          .many_ = many,
          .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
              n::parse_time("2019-05-01T01:00:00.000+02:00", "%FT%T%Ez"))},
          .maxTravelTime_ = 1800,
          .maxMatchingDistance_ = 250.0,
          .arriveBy_ = false,
          .useRoutedTransfers_ = true,
          .pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR,
          .maxPostTransitTime_ = 360});  // Too short to reach from U4

      EXPECT_EQ((std::vector<api::Duration>{
                    {720.0}, {780.0}, {} /* FIXME */, {960.0}, {}}),
                durations);
    }

    {
      // Test that location is reachable from FFM_HAUPT_S after arrival
      auto const test_durations =
          one_to_many_post(api::OneToManyIntermodalParams{
              .one_ = "50.10739,8.66333,-3",  // Near FFM_101
              .many_ = many,
              .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
                  n::parse_time("2019-05-01T01:00:00.000+02:00", "%FT%T%Ez"))},
              .maxTravelTime_ = 1800,
              .maxMatchingDistance_ = 250.0,
              .arriveBy_ = false,
              .useRoutedTransfers_ = true,
              .pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR,
              .maxPostTransitTime_ = 360});  // Reachable from S3
      EXPECT_EQ((std::vector<api::Duration>{
                    {} /* FIXME */, {1620.0}, {1320.0}, {1380.0}, {1680.0}}),
                test_durations);
    }

    {
      // Ensure all arriving stations are reachable
      // Also check that the correct arrival time is used
      auto const walk_durations =
          one_to_many_post(api::OneToManyIntermodalParams{
              .one_ = "50.107066,8.663604,0",
              .many_ = many,
              .time_ = {std::chrono::time_point_cast<std::chrono::seconds>(
                  n::parse_time("2019-05-01T01:00:00.000+02:00", "%FT%T%Ez"))},
              .maxTravelTime_ = 1800,
              .maxMatchingDistance_ = 250.0,
              .arriveBy_ = false,
              .useRoutedTransfers_ = true,
              .pedestrianProfile_ = api::PedestrianProfileEnum::FOOT,
              .maxPostTransitTime_ = 240});  // Only reachable from S3
      EXPECT_EQ((std::vector<api::Duration>{
                    {720.0},
                    {780.0},
                    {720.0},
                    {780.0},
                    {960.0},  // FIXME Must start at FFM_HAUPT_S => 1380 | 1500
                }),
                walk_durations);
    }
  }

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

  // TODO Delete after
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
    EXPECT_EQ("test_DA_10", ice.tripFrom_.stopId_);
    EXPECT_EQ("test_FFM_12", ice.tripTo_.stopId_);
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
    EXPECT_EQ("test_FFM_101", sbahn.tripFrom_.stopId_);
    EXPECT_EQ("test_FFM_10", sbahn.tripTo_.stopId_);
    EXPECT_EQ("S3", sbahn.displayName_);
    EXPECT_EQ("FFM Hbf", sbahn.headsign_);
    EXPECT_EQ("S3", sbahn.routeId_);
    EXPECT_EQ("2019-04-30 23:20", format_time(sbahn.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 23:20",
              format_time(sbahn.place_.scheduledArrival_.value()));
    EXPECT_EQ(false, sbahn.realTime_);
    EXPECT_EQ(2, sbahn.previousStops_->size());
  }

  {
    // same test with alerts off
    auto const res2 = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true"
        "&withAlerts=false");
    EXPECT_EQ(3, res2.stopTimes_.size());
    for (auto const& stopTime : res2.stopTimes_) {
      EXPECT_FALSE(stopTime.place_.alerts_.has_value());
    }
  }
}
