#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "fmt/format.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

#include "./util.h"

using namespace std::string_view_literals;
using namespace std::chrono_literals;
using namespace date;
using namespace motis;
namespace n = nigiri;

namespace {

// Three flex trips from A to B on 2019-05-01, agency timezone as parameter.
// Booking days of `br_service_day` only count S_BOOK days.
constexpr auto const kFlexGtfsTemplate = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,{}

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,50.1062497,8.6582223
B,Beispielberg Bahnhof,50.1079161,8.6605383

# location_groups.txt
location_group_id,location_group_name
lg_1,Rufbus 1
lg_2,Rufbus 2
lg_3,Rufbus 3

# location_group_stops.txt
location_group_id,stop_id
lg_1,A
lg_1,B
lg_2,A
lg_2,B
lg_3,A
lg_3,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_1,RUF,,Rufbus 1,715
FLEX_2,RUF,,Rufbus 2,715
FLEX_3,RUF,,Rufbus 3,715

# trips.txt
route_id,service_id,trip_id
FLEX_1,S1,flex_trip_1
FLEX_2,S1,flex_trip_2
FLEX_3,S1,flex_trip_3

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
S_BOOK,20190426,1
S_BOOK,20190429,1

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max,prior_notice_last_day,prior_notice_last_time,prior_notice_start_day,prior_notice_start_time,prior_notice_service_id,message,phone_number,booking_url
br_same_day,1,60,1440,,,,,,Book at least 1 hour in advance.,+43 123 456 789,https://example.org/book
br_prior_day,2,,,1,17:00:00,7,08:00:00,,Book until 17:00 the day before.,,
br_real_time,0,,,,,,,,,,
br_same_day_unbounded,1,30,,,,,,,,,
br_service_day,2,,,1,12:00:00,2,09:00:00,S_BOOK,,,

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type,pickup_booking_rule_id,drop_off_booking_rule_id
flex_trip_1,1,lg_1,08:30:00,18:00:00,2,1,br_prior_day,
flex_trip_1,2,lg_1,08:30:00,18:00:00,1,2,,br_same_day
flex_trip_2,1,lg_2,08:30:00,18:00:00,2,1,br_real_time,
flex_trip_2,2,lg_2,08:30:00,18:00:00,1,2,,br_same_day_unbounded
flex_trip_3,1,lg_3,08:30:00,18:00:00,2,1,br_service_day,
flex_trip_3,2,lg_3,08:30:00,18:00:00,1,2,,
)";

// Daily flex trip from A to B across the end of DST (2019-10-27).
constexpr auto const kFlexDstGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,50.1062497,8.6582223
B,Beispielberg Bahnhof,50.1079161,8.6605383

# location_groups.txt
location_group_id,location_group_name
lg_1,Rufbus 1

# location_group_stops.txt
location_group_id,stop_id
lg_1,A
lg_1,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_1,RUF,,Rufbus 1,715

# trips.txt
route_id,service_id,trip_id
FLEX_1,S1,flex_trip_1

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S1,1,1,1,1,1,1,1,20191025,20191030

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_last_day,prior_notice_last_time
br_prior_day,2,1,17:00:00

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type,pickup_booking_rule_id,drop_off_booking_rule_id
flex_trip_1,1,lg_1,08:30:00,18:00:00,2,1,br_prior_day,
flex_trip_1,2,lg_1,08:30:00,18:00:00,1,2,,br_prior_day
)"sv;

config flex_config(std::string_view const gtfs,
                   std::string const& first_day,
                   std::uint16_t const num_days) {
  return config{.osm_ = {"test/resources/test_case.osm.pbf"},
                .timetable_ =
                    config::timetable{
                        .first_day_ = first_day,
                        .num_days_ = num_days,
                        .datasets_ = {{"test", {.path_ = std::string{gtfs}}}}},
                .street_routing_ = true};
}

api::plan_response route_a_to_b(ep::routing const& routing,
                                sys_seconds const time) {
  return routing(
      "?fromPlace=50.1062497,8.6582223"
      "&toPlace=50.1079161,8.6605383"
      "&time=" +
      format("%FT%H:%MZ", time) +
      "&timetableView=false"
      "&transitModes="
      "&directModes=FLEX");
}

// The FLEX leg of the trip with the given location group.
api::Leg const* get_flex_leg(api::plan_response const& res,
                             std::string_view const flex_id) {
  for (auto const& it : res.direct_) {
    for (auto const& l : it.legs_) {
      if (l.mode_ == api::ModeEnum::FLEX && l.from_.flexId_ == flex_id) {
        EXPECT_EQ("Musterdorf Hauptplatz", l.from_.name_);
        EXPECT_EQ("Beispielberg Bahnhof", l.to_.name_);
        return &l;
      }
    }
  }
  return nullptr;
}

}  // namespace

TEST(motis, flex_booking_rules) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_booking_rules", ec);

  auto const c = flex_config(fmt::format(kFlexGtfsTemplate, "Europe/Berlin"),
                             "2019-04-25", 8U);
  import(c, "test/data_flex_booking_rules");
  auto d = data{"test/data_flex_booking_rules", c};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = route_a_to_b(routing, sys_days{2019_y / May / 1} + 7h);

  // UP_TO_PRIOR_DAYS: from 7 days before at 08:00 until 1 day before at 17:00.
  auto const* trip_1 = get_flex_leg(res, "lg_1");
  ASSERT_NE(nullptr, trip_1);
  ASSERT_TRUE(trip_1->from_.pickupBookingRule_.has_value());
  auto const& prior_day = *trip_1->from_.pickupBookingRule_;
  EXPECT_EQ(api::BookingTypeEnum::UP_TO_PRIOR_DAYS, prior_day.bookingType_);
  EXPECT_EQ(sys_days{2019_y / April / 24} + 6h,
            prior_day.earliestBookingTime_.value().time_);
  EXPECT_EQ(sys_days{2019_y / April / 30} + 15h,
            prior_day.latestBookingTime_.value().time_);
  EXPECT_EQ("Book until 17:00 the day before.", prior_day.message_);

  // UP_TO_SAME_DAY: relative to the start of the leg.
  ASSERT_TRUE(trip_1->to_.dropOffBookingRule_.has_value());
  auto const& same_day = *trip_1->to_.dropOffBookingRule_;
  EXPECT_EQ(api::BookingTypeEnum::UP_TO_SAME_DAY, same_day.bookingType_);
  EXPECT_EQ(60, same_day.priorNoticeDurationMin_);
  EXPECT_EQ(1440, same_day.priorNoticeDurationMax_);
  EXPECT_EQ(trip_1->startTime_.time_ - 1440min,
            same_day.earliestBookingTime_.value().time_);
  EXPECT_EQ(trip_1->startTime_.time_ - 60min,
            same_day.latestBookingTime_.value().time_);
  EXPECT_EQ("+43 123 456 789", same_day.phoneNumber_);
  EXPECT_EQ("https://example.org/book", same_day.bookingUrl_);

  // REAL_TIME: no booking times.
  auto const* trip_2 = get_flex_leg(res, "lg_2");
  ASSERT_NE(nullptr, trip_2);
  ASSERT_TRUE(trip_2->from_.pickupBookingRule_.has_value());
  auto const& real_time = *trip_2->from_.pickupBookingRule_;
  EXPECT_EQ(api::BookingTypeEnum::REAL_TIME, real_time.bookingType_);
  EXPECT_FALSE(real_time.earliestBookingTime_.has_value());
  EXPECT_FALSE(real_time.latestBookingTime_.has_value());

  // UP_TO_SAME_DAY without maximum notice: no earliest booking time.
  ASSERT_TRUE(trip_2->to_.dropOffBookingRule_.has_value());
  auto const& unbounded = *trip_2->to_.dropOffBookingRule_;
  EXPECT_FALSE(unbounded.priorNoticeDurationMax_.has_value());
  EXPECT_FALSE(unbounded.earliestBookingTime_.has_value());
  EXPECT_EQ(trip_2->startTime_.time_ - 30min,
            unbounded.latestBookingTime_.value().time_);

  // prior_notice_service_id: 1 and 2 S_BOOK days before are 04-29 and 04-26.
  auto const* trip_3 = get_flex_leg(res, "lg_3");
  ASSERT_NE(nullptr, trip_3);
  ASSERT_TRUE(trip_3->from_.pickupBookingRule_.has_value());
  auto const& service_day = *trip_3->from_.pickupBookingRule_;
  EXPECT_EQ(sys_days{2019_y / April / 26} + 7h,
            service_day.earliestBookingTime_.value().time_);
  EXPECT_EQ(sys_days{2019_y / April / 29} + 10h,
            service_day.latestBookingTime_.value().time_);
  EXPECT_FALSE(trip_3->to_.dropOffBookingRule_.has_value());
}

TEST(motis, flex_booking_rules_dst) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_booking_rules_dst", ec);

  auto const c = flex_config(kFlexDstGtfs, "2019-10-25", 6U);
  import(c, "test/data_flex_booking_rules_dst");
  auto d = data{"test/data_flex_booking_rules_dst", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  // Deadline: 17:00 local time on the day before the day of travel.
  // CEST = UTC+2 until 2019-10-27 03:00, then CET = UTC+1.
  for (auto const& [travel, deadline] :
       {std::pair{sys_days{2019_y / October / 26},
                  sys_days{2019_y / October / 25} + 15h},
        std::pair{sys_days{2019_y / October / 27},
                  sys_days{2019_y / October / 26} + 15h},
        std::pair{sys_days{2019_y / October / 28},
                  sys_days{2019_y / October / 27} + 16h},
        std::pair{sys_days{2019_y / October / 29},
                  sys_days{2019_y / October / 28} + 16h}}) {
    auto const res = route_a_to_b(routing, travel + 7h);
    auto const* leg = get_flex_leg(res, "lg_1");
    ASSERT_NE(nullptr, leg) << travel;
    ASSERT_TRUE(leg->from_.pickupBookingRule_.has_value()) << travel;
    EXPECT_EQ(deadline,
              leg->from_.pickupBookingRule_->latestBookingTime_.value().time_)
        << travel;
  }
}

TEST(motis, flex_booking_rules_local_day) {
  // The day of travel 2019-05-01 is a different UTC day:
  // 09:00 NZST (UTC+12) = 04-30 21:00Z, 17:00 HST (UTC-10) = 05-02 03:00Z.
  // The time window 08:30-18:00 local starts on the UTC day before (NZST) or
  // ends on the UTC day after (HST).
  struct expected {
    char const* tz_;
    sys_seconds travel_;
    sys_seconds window_from_, window_to_;
    sys_seconds prior_day_earliest_, prior_day_latest_;
    sys_seconds service_day_earliest_, service_day_latest_;
  };
  for (auto const& x :
       {expected{.tz_ = "Pacific/Auckland",
                 .travel_ = sys_days{2019_y / April / 30} + 21h,
                 .window_from_ = sys_days{2019_y / April / 30} + 20h + 30min,
                 .window_to_ = sys_days{2019_y / May / 1} + 6h,
                 .prior_day_earliest_ = sys_days{2019_y / April / 23} + 20h,
                 .prior_day_latest_ = sys_days{2019_y / April / 30} + 5h,
                 .service_day_earliest_ = sys_days{2019_y / April / 25} + 21h,
                 .service_day_latest_ = sys_days{2019_y / April / 29}},
        expected{.tz_ = "Pacific/Honolulu",
                 .travel_ = sys_days{2019_y / May / 2} + 3h,
                 .window_from_ = sys_days{2019_y / May / 1} + 18h + 30min,
                 .window_to_ = sys_days{2019_y / May / 2} + 4h,
                 .prior_day_earliest_ = sys_days{2019_y / April / 24} + 18h,
                 .prior_day_latest_ = sys_days{2019_y / May / 1} + 3h,
                 .service_day_earliest_ = sys_days{2019_y / April / 26} + 19h,
                 .service_day_latest_ = sys_days{2019_y / April / 29} + 22h}}) {
    auto const path = fmt::format("test/data_flex_booking_rules_{}",
                                  std::string_view{x.tz_}.substr(8));
    auto ec = std::error_code{};
    std::filesystem::remove_all(path, ec);

    auto const c =
        flex_config(fmt::format(kFlexGtfsTemplate, x.tz_), "2019-04-25", 8U);
    import(c, path);
    auto d = data{path, c};
    auto const routing = utl::init_from<ep::routing>(d).value();
    auto const res = route_a_to_b(routing, x.travel_);

    auto const* trip_1 = get_flex_leg(res, "lg_1");
    ASSERT_NE(nullptr, trip_1) << x.tz_;
    for (auto const* p : {&trip_1->from_, &trip_1->to_}) {
      EXPECT_EQ(x.window_from_, p->flexStartPickupDropOffWindow_.value().time_)
          << x.tz_;
      EXPECT_EQ(x.window_to_, p->flexEndPickupDropOffWindow_.value().time_)
          << x.tz_;
    }

    ASSERT_TRUE(trip_1->from_.pickupBookingRule_.has_value()) << x.tz_;
    auto const& prior_day = *trip_1->from_.pickupBookingRule_;
    EXPECT_EQ(x.prior_day_earliest_,
              prior_day.earliestBookingTime_.value().time_)
        << x.tz_;
    EXPECT_EQ(x.prior_day_latest_, prior_day.latestBookingTime_.value().time_)
        << x.tz_;

    auto const* trip_3 = get_flex_leg(res, "lg_3");
    ASSERT_NE(nullptr, trip_3) << x.tz_;
    ASSERT_TRUE(trip_3->from_.pickupBookingRule_.has_value()) << x.tz_;
    auto const& service_day = *trip_3->from_.pickupBookingRule_;
    EXPECT_EQ(x.service_day_earliest_,
              service_day.earliestBookingTime_.value().time_)
        << x.tz_;
    EXPECT_EQ(x.service_day_latest_,
              service_day.latestBookingTime_.value().time_)
        << x.tz_;
  }
}

TEST(motis, flex_alerts) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_alerts", ec);

  auto const c = flex_config(fmt::format(kFlexGtfsTemplate, "Europe/Berlin"),
                             "2019-04-25", 8U);
  import(c, "test/data_flex_alerts");
  auto d = data{"test/data_flex_alerts", c};
  d.init_rtt(sys_days{2019_y / May / 1});

  auto msg =
      test::to_feed_msg({test::alert{.header_ = "Booking required",
                                     .description_ = "Book via the Rufbus app.",
                                     .entities_ = {{.route_id_ = "FLEX_1"}}},
                         test::alert{.header_ = "Stop moved",
                                     .description_ = "Use the other side.",
                                     .entities_ = {{.stop_id_ = "A"}}}},
                        sys_days{2019_y / May / 1} + 6h);
  msg.mutable_entity(0)->mutable_alert()->set_severity_level(
      transit_realtime::Alert_SeverityLevel_WARNING);
  n::rt::gtfsrt_update_msg(*d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
                           msg);

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = route_a_to_b(routing, sys_days{2019_y / May / 1} + 7h);

  // Route alert on the leg, stop alert on the place.
  for (auto const flex_id : {"lg_1"sv, "lg_2"sv}) {
    auto const* leg = get_flex_leg(res, flex_id);
    ASSERT_NE(nullptr, leg);
    ASSERT_TRUE(leg->from_.alerts_.has_value());
    ASSERT_EQ(1U, leg->from_.alerts_->size());
    EXPECT_EQ("Stop moved", leg->from_.alerts_->front().headerText_);
    EXPECT_FALSE(leg->to_.alerts_.has_value());
    if (flex_id == "lg_1") {
      ASSERT_TRUE(leg->alerts_.has_value());
      ASSERT_EQ(1U, leg->alerts_->size());
      EXPECT_EQ("Booking required", leg->alerts_->front().headerText_);
      EXPECT_EQ(api::AlertSeverityLevelEnum::WARNING,
                leg->alerts_->front().severityLevel_);
    } else {
      EXPECT_FALSE(leg->alerts_.has_value());
    }
  }
}
