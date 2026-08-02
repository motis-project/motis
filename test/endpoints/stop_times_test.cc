#include "motis/endpoints/stop_times.h"
#include "gtest/gtest.h"

#include <chrono>
#include <sstream>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/json.hpp"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/frun.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

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
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code,stop_code
DA,DA Hbf,49.87260,8.63085,1,,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10,DA-10-CODE
FFM,FFM Hbf,50.10701,8.66341,1,,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10,
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12,
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,,U4,
LANGEN,Langen,49.99359,8.65677,1,,1,
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,,

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
  import(c, "test/data");
  auto d = data{"test/data", c};
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

  auto const format_time = [](auto&& t, char const* fmt = "%F %H:%M") {
    return date::format(fmt, *t);
  };

  {
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true");

    EXPECT_EQ("test_FFM_10", res.place_.stopId_);
    EXPECT_EQ(3, res.stopTimes_.size());

    auto const& ice = res.stopTimes_[0];
    EXPECT_EQ(api::ModeEnum::HIGHSPEED_RAIL, ice.mode_);
    EXPECT_EQ("20190501_00:35_test_ICE", ice.tripId_);
    EXPECT_EQ("test_DA_10", ice.tripFrom_.stopId_);
    EXPECT_EQ("DA-10-CODE", ice.tripFrom_.stopCode_);
    EXPECT_FALSE(ice.tripTo_.stopCode_.has_value());
    EXPECT_EQ("test_FFM_12", ice.tripTo_.stopId_);
    EXPECT_EQ("ICE", ice.displayName_);
    EXPECT_EQ("FFM Hbf", ice.headsign_);
    EXPECT_EQ("test_ICE", ice.routeId_);
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
    EXPECT_EQ("test_S3", sbahn.routeId_);
    EXPECT_EQ("2019-04-30 23:20", format_time(sbahn.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 23:20",
              format_time(sbahn.place_.scheduledArrival_.value()));
    EXPECT_EQ(false, sbahn.realTime_);
    EXPECT_EQ(2, sbahn.previousStops_->size());
    EXPECT_FALSE(res.predictionDebug_.has_value());

    auto const [ice_run, _] =
        d.tags_->get_trip(*d.tt_, d.rt_->rtt_.get(), ice.tripId_);
    auto const ice_frun = n::rt::frun{*d.tt_, d.rt_->rtt_.get(), ice_run};
    auto const scheduled =
        std::chrono::duration_cast<std::chrono::seconds>(
            static_cast<std::chrono::sys_seconds>(
                *ice.place_.scheduledArrival_)
                .time_since_epoch())
            .count();
    d.rt_->vehicle_prediction_diagnostics_ =
        vehicle_prediction_diagnostics_store::build(
            true,
            {{.transport_ = ice_frun.t_,
              .static_stop_sequence_ = 2U,
              .trip_id_ = ice.tripId_,
              .observed_at_seconds_ = scheduled,
              .provider_ = prediction_candidate_diagnostic{
                  .source_ = vehicle_prediction_source::kProvider,
                  .predicted_timestamp_seconds_ = scheduled + 600,
                  .delay_seconds_ = 600},
              .gps_ = prediction_candidate_diagnostic{
                  .source_ = vehicle_prediction_source::kGps,
                  .predicted_timestamp_seconds_ = scheduled + 420,
                  .delay_seconds_ = 420,
                  .confidence_ = 0.8},
              .effective_ = {.source_ =
                                 vehicle_prediction_source::kProvider,
                             .predicted_timestamp_seconds_ = scheduled + 600,
                             .delay_seconds_ = 600},
              .selected_source_ = vehicle_prediction_source::kGps,
              .selection_reason_ = vehicle_prediction_selection_reason::
                  kProviderProgressInconsistent}},
            scheduled);

    auto const debug_res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&includePredictionComparison=true");
    ASSERT_TRUE(debug_res.predictionDebug_.has_value());
    ASSERT_EQ(1U, debug_res.predictionDebug_->size());
    auto const& debug = debug_res.predictionDebug_->front();
    EXPECT_EQ(0, debug.stopTimeIndex_);
    EXPECT_EQ(ice.tripId_, debug.tripId_);
    EXPECT_EQ(420, debug.gps_->delaySeconds_);
    EXPECT_EQ(600, debug.effective_.delaySeconds_);
    EXPECT_EQ(api::PredictionSourceEnum::GPS, debug.selectedSource_);
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

  {
    // center-only query, radius is required
    auto const res = stop_times(
        "/api/v5/stoptimes?center=50.10593,8.66118"
        "&radius=250"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true");

    EXPECT_EQ("center", res.place_.name_);
    EXPECT_FALSE(res.place_.stopId_.has_value());
    EXPECT_FALSE(res.stopTimes_.empty());
  }

  {
    // invalid stopId without center
    EXPECT_THROW(stop_times("/api/v5/stoptimes?stopId=test_SOMETHING_RANDOM"
                            "&time=2019-04-30T23:30:00.000Z"
                            "&arriveBy=true"
                            "&n=3"),
                 net::not_found_exception);
  }

  {
    // invalid stopId should fall back to center
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_SOMETHING_RANDOM"
        "&center=50.10593,8.66118"
        "&radius=250"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de");

    EXPECT_EQ("center", res.place_.name_);
    EXPECT_FALSE(res.place_.stopId_.has_value());
    EXPECT_FALSE(res.stopTimes_.empty());
  }

  {
    // stoptimes in radius = r
    auto const r = 110.0;
    auto const center = geo::latlng{50.10563, 8.66218};
    auto const res =
        stop_times(std::format("/api/v5/stoptimes?center={},{}"
                               "&radius={}"
                               "&exactRadius=true"
                               "&time=2019-04-30T23:30:00.000Z"
                               "&arriveBy=true"
                               "&n=200"
                               "&language=de"
                               "&fetchStops=true",
                               center.lat_, center.lng_, r));
    EXPECT_FALSE(res.stopTimes_.empty());
    for (auto const& v : res.stopTimes_) {
      auto const dist =
          geo::distance(center, geo::latlng{v.place_.lat_, v.place_.lon_});
      EXPECT_LE(dist, r);
    }
  }

  {
    // neither stopId nor center -> panic
    EXPECT_THROW(stop_times("/api/v5/stoptimes?time=2019-04-30T23:30:00.000Z"
                            "&arriveBy=true"
                            "&n=3"),
                 net::bad_request_exception);
  }

  {
    // center without stopId requires radius
    EXPECT_THROW(stop_times("/api/v5/stoptimes?center=50.10593,8.66118"
                            "&time=2019-04-30T23:30:00.000Z"
                            "&arriveBy=true"
                            "&n=3"),
                 net::bad_request_exception);
  }

  {
    // window query LATER
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_101"
        "&time=2019-04-30T23:00:00.000Z"
        "&arriveBy=true"
        "&direction=LATER"
        "&window=1800"
        "&language=de");

    EXPECT_EQ(2, res.stopTimes_.size());  // n is ignored if window is set
    for (auto const& stop_time : res.stopTimes_) {
      auto const arr = format_time(stop_time.place_.arrival_.value());
      std::cout << "arr: " << arr << std::endl;
      EXPECT_GE(arr, "2019-04-30 23:00");
      EXPECT_LE(arr, "2019-04-30 23:30");
    }
    EXPECT_FALSE(res.previousPageCursor_.empty());
    EXPECT_FALSE(res.nextPageCursor_.empty());
  }
  {
    // window query EARLIER
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_101"
        "&time=2019-04-30T23:15:00.000Z"
        "&arriveBy=true"
        "&direction=EARLIER"
        "&window=1800"
        "&language=de");

    for (auto const& stop_time : res.stopTimes_) {
      auto const arr = format_time(stop_time.place_.arrival_.value());
      std::cout << "arr E: " << arr << std::endl;
      EXPECT_GE(arr, "2019-04-30 22:45");
      EXPECT_LE(arr, "2019-04-30 23:15");
    }
  }
  {
    // window query EARLIER (small window large n)
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_101"
        "&time=2019-04-30T23:15:00.000Z"
        "&arriveBy=true"
        "&direction=LATER"
        "&window=60"
        "&n=2"
        "&language=de");

    EXPECT_GT(res.stopTimes_.size(), 1);
    for (auto const& stop_time : res.stopTimes_) {
      auto const arr = format_time(stop_time.place_.arrival_.value());
      std::cout << "arr E2: " << arr << std::endl;
    }
  }

  {
    // realtimeMode=OFF:
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true"
        "&realtimeMode=OFF");

    auto const& ice = res.stopTimes_[0];
    EXPECT_EQ("20190501_00:35_test_ICE", ice.tripId_);
    // No realtime:
    EXPECT_EQ("test_FFM_10", ice.tripTo_.stopId_);
    EXPECT_EQ(false, ice.realTime_);
    EXPECT_EQ("2019-04-30 22:45", format_time(ice.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 22:45",
              format_time(ice.place_.scheduledArrival_.value()));
    // No realtime alerts either
    EXPECT_FALSE(ice.place_.alerts_.has_value() &&
                 !ice.place_.alerts_->empty());
  }

  {
    // realtimeMode=REALTIME_ANNOTATION_ONLY:
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_FFM_10"
        "&time=2019-04-30T23:30:00.000Z"
        "&arriveBy=true"
        "&n=3"
        "&language=de"
        "&fetchStops=true"
        "&realtimeMode=REALTIME_ANNOTATION_ONLY");

    auto const& ice = res.stopTimes_[0];
    EXPECT_EQ("20190501_00:35_test_ICE", ice.tripId_);
    // Annotated with realtime:
    EXPECT_EQ(true, ice.realTime_);
    EXPECT_EQ("2019-04-30 22:55", format_time(ice.place_.arrival_.value()));
    EXPECT_EQ("2019-04-30 22:45",
              format_time(ice.place_.scheduledArrival_.value()));
    // Realtime alert is annotated too
    EXPECT_EQ(1, ice.place_.alerts_->size());
  }
}

// T1 and T2 share block_id "block_1" (T2 continues where T1 ends), so they
// are combined into a single transport/run. T1 requires a compulsory
// reservation (pickup_type=2 on trips.txt), T2 does not (default clasz BUS
// is not compulsory by default) -- the reservation flag is per-section, so
// it must be reported correctly for each leg of the combined run.
constexpr auto const kGTFS_RESERVATION = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.10701,8.66341
B,B,50.11403,8.67835
C,C,49.87260,8.63085

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,pickup_type
R1,S1,T1,,block_1,2
R2,S1,T2,,block_1,0

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,08:00:00,08:00:00,A,0
T1,08:10:00,08:10:00,B,1
T2,08:10:00,08:10:00,B,0
T2,08:20:00,08:20:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, stop_times_reservation) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data-reservation", ec);

  auto const c =
      config{.timetable_ = config::timetable{
                 .first_day_ = "2019-05-01",
                 .num_days_ = 2,
                 .datasets_ = {{"test", {.path_ = kGTFS_RESERVATION}}}}};
  import(c, "test/data-reservation");
  auto d = data{"test/data-reservation", c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  {
    // Departure at A: first leg of the combined run (T1), compulsory
    // reservation. tripFrom_/tripTo_ are T1's own (sub-trip) endpoints,
    // while nextStops_ spans the whole combined run, i.e. it reaches past
    // T1's own destination (B) into T2's leg (C).
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_A"
        "&time=2019-05-01T05:55:00.000Z"
        "&n=1"
        "&language=de"
        "&fetchStops=true");

    ASSERT_EQ(1, res.stopTimes_.size());
    auto const& t1 = res.stopTimes_[0];
    EXPECT_EQ("20190501_08:00_test_T1", t1.tripId_);
    EXPECT_EQ(api::ReservationEnum::COMPULSORY, t1.reservation_);
    EXPECT_EQ("test_A", t1.tripFrom_.stopId_);
    EXPECT_EQ("test_B", t1.tripTo_.stopId_);  // T1's own destination
    ASSERT_EQ(2, t1.nextStops_->size());
    EXPECT_EQ("test_B", (*t1.nextStops_)[0].stopId_);
    EXPECT_EQ("test_C", (*t1.nextStops_)[1].stopId_);  // reaches into T2's leg
  }

  {
    // Arrival at C: second leg of the combined run (T2), no reservation
    // required. previousStops_ spans the whole combined run, reaching back
    // past T2's own origin (B) into T1's leg (A), proving both trips were
    // merged into a single transport via block_id.
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_C"
        "&time=2019-05-01T06:25:00.000Z"
        "&arriveBy=true"
        "&n=1"
        "&language=de"
        "&fetchStops=true");

    ASSERT_EQ(1, res.stopTimes_.size());
    auto const& t2 = res.stopTimes_[0];
    EXPECT_EQ("20190501_08:10_test_T2", t2.tripId_);
    EXPECT_EQ(api::ReservationEnum::NONE, t2.reservation_);
    EXPECT_EQ("test_B", t2.tripFrom_.stopId_);  // T2's own origin
    EXPECT_EQ("test_C", t2.tripTo_.stopId_);
    ASSERT_EQ(2, t2.previousStops_->size());
    EXPECT_EQ("test_A",
              (*t2.previousStops_)[0].stopId_);  // reaches into T1's leg
    EXPECT_EQ("test_B", (*t2.previousStops_)[1].stopId_);
  }
}

// Service S1 runs daily from 2019-05-01 to 2019-05-10 (feed_end_date).
// With extend_calendar=true, the calendar loops beyond that date, so a
// trip on 2019-05-20 is a "looped" trip since 2019-05-10.
constexpr auto const kGTFS_LOOPED_CALENDAR = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.10701,8.66341
B,B,50.11403,8.67835

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
R1,S1,T1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,08:00:00,08:00:00,A,0
T1,08:10:00,08:10:00,B,1

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S1,1,1,1,1,1,1,1,20190501,20190510

# feed_info.txt
feed_publisher_name,feed_publisher_url,feed_lang,feed_end_date
Test,https://example.com,en,20190510
)";

TEST(motis, stop_times_looped_calendar_since) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data-looped-calendar", ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 30,
                            .datasets_ = {{"test",
                                           {.path_ = kGTFS_LOOPED_CALENDAR,
                                            .extend_calendar_ = true}}}}};
  import(c, "test/data-looped-calendar");
  auto d = data{"test/data-looped-calendar", c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  auto const format_time = [](auto&& t, char const* fmt = "%F %H:%M") {
    return date::format(fmt, *t);
  };

  {
    // Before the feed's end date: no looped calendar.
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_A"
        "&time=2019-05-01T05:55:00.000Z"
        "&n=1"
        "&language=de");

    ASSERT_EQ(1, res.stopTimes_.size());
    EXPECT_EQ("20190501_08:00_test_T1", res.stopTimes_[0].tripId_);
    EXPECT_FALSE(res.stopTimes_[0].loopedCalendarSince_.has_value());
  }

  {
    // After the feed's end date: the calendar has looped since
    // 2019-05-10 (feed_end_date).
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_A"
        "&time=2019-05-20T05:55:00.000Z"
        "&n=1"
        "&language=de");

    ASSERT_EQ(1, res.stopTimes_.size());
    EXPECT_EQ("20190520_08:00_test_T1", res.stopTimes_[0].tripId_);
    ASSERT_TRUE(res.stopTimes_[0].loopedCalendarSince_.has_value());
    EXPECT_EQ("2019-05-10 00:00",
              format_time(res.stopTimes_[0].loopedCalendarSince_.value()));
  }
}
