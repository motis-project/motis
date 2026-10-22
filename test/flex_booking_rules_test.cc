#include "gtest/gtest.h"

#include <chrono>
#include <string>

#include "boost/json.hpp"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/flex/flex_output.h"
#include "motis/import.h"

#include "./util.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace std::chrono_literals;
using namespace date;
using namespace motis;
namespace n = nigiri;

namespace {

constexpr auto const kFlexGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,49.87526849014631,8.62771903392948
B,Beispielberg Bahnhof,49.87253873915287,8.629724234688751

# location_groups.txt
location_group_id,location_group_name
lg_1,Rufbus Musterland
lg_2,Rufbus Musterland Express
lg_3,Rufbus Musterland Werktag

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
FLEX_1,RUF,,Rufbus Musterland,715
FLEX_2,RUF,,Rufbus Musterland Express,715
FLEX_3,RUF,,Rufbus Musterland Werktag,715

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
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max,prior_notice_last_day,prior_notice_last_time,prior_notice_start_day,prior_notice_start_time,prior_notice_service_id,message,pickup_message,drop_off_message,phone_number,info_url,booking_url
br_same_day,1,60,1440,,,,,,Book at least 1 hour in advance.,,,+43 123 456 789,https://example.org/info,https://example.org/book
br_prior_day,2,,,1,17:00:00,7,08:00:00,,Book until 17:00 the day before.,Pickup only after booking.,,+43 123 456 789,,
br_real_time,0,,,,,,,,,,,,,https://example.org/book
br_same_day_unbounded,1,30,,,,,,,,,,,,
br_prior_service_day,2,,,1,12:00:00,2,09:00:00,S_BOOK,,,,,,

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type,pickup_booking_rule_id,drop_off_booking_rule_id
flex_trip_1,1,lg_1,08:30:00,18:00:00,2,1,br_prior_day,
flex_trip_1,2,lg_1,08:30:00,18:00:00,1,2,,br_same_day
flex_trip_2,1,lg_2,08:30:00,18:00:00,2,1,br_real_time,
flex_trip_2,2,lg_2,08:30:00,18:00:00,1,2,,br_same_day_unbounded
flex_trip_3,1,lg_3,08:30:00,18:00:00,2,1,br_prior_service_day,
flex_trip_3,2,lg_3,08:30:00,18:00:00,1,2,,
)"sv;

constexpr auto const kFlexNoBookingRulesGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,49.87526849014631,8.62771903392948
B,Beispielberg Bahnhof,49.87253873915287,8.629724234688751

# location_groups.txt
location_group_id,location_group_name
lg_1,Rufbus Musterland

# location_group_stops.txt
location_group_id,stop_id
lg_1,A
lg_1,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_1,RUF,,Rufbus Musterland,715

# trips.txt
route_id,service_id,trip_id
FLEX_1,S1,flex_trip_1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
flex_trip_1,1,lg_1,08:30:00,18:00:00,2,1
flex_trip_1,2,lg_1,08:30:00,18:00:00,1,2
)"sv;

constexpr auto const kFlexQuery =
    "?fromPlace=49.8760,8.6270"
    "&toPlace=49.8718,8.6305"
    "&time=2019-05-01T07:00Z"
    "&timetableView=false"
    "&transitModes="
    "&directModes=FLEX";

// Daily flex trip across the end of DST (2019-10-27 in Europe/Berlin).
constexpr auto const kFlexDstGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,49.87526849014631,8.62771903392948
B,Beispielberg Bahnhof,49.87253873915287,8.629724234688751

# location_groups.txt
location_group_id,location_group_name
lg_1,Rufbus Musterland

# location_group_stops.txt
location_group_id,stop_id
lg_1,A
lg_1,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_1,RUF,,Rufbus Musterland,715

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

config flex_config(std::string_view gtfs,
                   std::string first_day = "2019-04-25",
                   std::uint16_t const num_days = 8U) {
  return config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
                .osm_ = {"test/resources/test_case.osm.pbf"},
                .timetable_ =
                    config::timetable{
                        .first_day_ = std::move(first_day),
                        .num_days_ = num_days,
                        .datasets_ = {{"test", {.path_ = std::string{gtfs}}}}},
                .street_routing_ = true};
}

api::Leg const* find_flex_leg(api::plan_response const& res,
                              std::string_view const flex_id) {
  for (auto const& it : res.direct_) {
    for (auto const& leg : it.legs_) {
      if (leg.mode_ == api::ModeEnum::FLEX && leg.from_.flexId_ == flex_id) {
        return &leg;
      }
    }
  }
  return nullptr;
}

}  // namespace

TEST(motis, flex_local_time_to_unixtime) {
  auto const tz = date::locate_zone("Europe/Berlin");
  auto const at = [&](date::local_days const day, n::duration_t const t) {
    return flex::local_time_to_unixtime(tz, day, t);
  };

  // 17:00 local: 15:00 UTC in summer time, 16:00 UTC in winter time.
  EXPECT_EQ(sys_days{2019_y / April / 30} + 15h,
            at(local_days{2019_y / April / 30}, 17h));
  EXPECT_EQ(sys_days{2019_y / October / 28} + 16h,
            at(local_days{2019_y / October / 28}, 17h));

  // DST starts on 2019-03-31 at 02:00 CET.
  EXPECT_EQ(sys_days{2019_y / March / 30} + 23h + 30min,  // 00:30 CET
            at(local_days{2019_y / March / 31}, 30min));
  EXPECT_EQ(sys_days{2019_y / March / 31} + 1h,  // 02:30 does not exist
            at(local_days{2019_y / March / 31}, 2h + 30min));
  EXPECT_EQ(sys_days{2019_y / March / 31} + 15h,  // 17:00 CEST
            at(local_days{2019_y / March / 31}, 17h));

  // DST ends on 2019-10-27 at 03:00 CEST.
  EXPECT_EQ(sys_days{2019_y / October / 26} + 22h + 30min,  // 00:30 CEST
            at(local_days{2019_y / October / 27}, 30min));
  EXPECT_EQ(sys_days{2019_y / October / 27} + 30min,  // 02:30 CEST (1st)
            at(local_days{2019_y / October / 27}, 2h + 30min));
  EXPECT_EQ(sys_days{2019_y / October / 27} + 16h,  // 17:00 CET
            at(local_days{2019_y / October / 27}, 17h));
}

TEST(motis, flex_booking_rules) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_booking_rules", ec);

  auto const c = flex_config(kFlexGTFS);
  import(c, "test/data_flex_booking_rules");
  auto d = data{"test/data_flex_booking_rules", c};

  // Import: booking rules reach the timetable.
  ASSERT_EQ(5U, d.tt_->booking_rules_.size());

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(kFlexQuery);

  auto const* trip_1 = find_flex_leg(res, "lg_1");
  auto const* trip_2 = find_flex_leg(res, "lg_2");
  auto const* trip_3 = find_flex_leg(res, "lg_3");
  ASSERT_NE(nullptr, trip_1);
  ASSERT_NE(nullptr, trip_2);
  ASSERT_NE(nullptr, trip_3);

  // Service day 2019-05-01, Europe/Berlin = UTC+2.

  // flex_trip_1: pickup = br_prior_day (UP_TO_PRIOR_DAYS)
  {
    ASSERT_TRUE(trip_1->from_.pickupBookingRule_.has_value());
    auto const& r = *trip_1->from_.pickupBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::UP_TO_PRIOR_DAYS, r.bookingType_);
    // 7 days before at 08:00 local, until 1 day before at 17:00 local.
    ASSERT_TRUE(r.earliestBookingTime_.has_value());
    ASSERT_TRUE(r.latestBookingTime_.has_value());
    EXPECT_EQ(sys_days{2019_y / April / 24} + 6h,
              r.earliestBookingTime_->time_);
    EXPECT_EQ(sys_days{2019_y / April / 30} + 15h, r.latestBookingTime_->time_);
    EXPECT_FALSE(r.priorNoticeDurationMin_.has_value());
    EXPECT_FALSE(r.priorNoticeDurationMax_.has_value());
    EXPECT_EQ("Book until 17:00 the day before.", r.message_);
    EXPECT_EQ("Pickup only after booking.", r.pickupMessage_);
    EXPECT_FALSE(r.dropOffMessage_.has_value());
    EXPECT_EQ("+43 123 456 789", r.phoneNumber_);
    EXPECT_FALSE(r.infoUrl_.has_value());
    EXPECT_FALSE(r.bookingUrl_.has_value());
  }
  EXPECT_FALSE(trip_1->from_.dropOffBookingRule_.has_value());
  EXPECT_FALSE(trip_1->to_.pickupBookingRule_.has_value());

  // flex_trip_1: drop off = br_same_day (UP_TO_SAME_DAY)
  {
    ASSERT_TRUE(trip_1->to_.dropOffBookingRule_.has_value());
    auto const& r = *trip_1->to_.dropOffBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::UP_TO_SAME_DAY, r.bookingType_);
    EXPECT_EQ(60, r.priorNoticeDurationMin_);
    EXPECT_EQ(1440, r.priorNoticeDurationMax_);
    // Relative to the start of the leg.
    ASSERT_TRUE(r.earliestBookingTime_.has_value());
    ASSERT_TRUE(r.latestBookingTime_.has_value());
    EXPECT_EQ(trip_1->startTime_.time_ - 1440min,
              r.earliestBookingTime_->time_);
    EXPECT_EQ(trip_1->startTime_.time_ - 60min, r.latestBookingTime_->time_);
    EXPECT_EQ("Book at least 1 hour in advance.", r.message_);
    EXPECT_FALSE(r.pickupMessage_.has_value());
    EXPECT_EQ("+43 123 456 789", r.phoneNumber_);
    EXPECT_EQ("https://example.org/info", r.infoUrl_);
    EXPECT_EQ("https://example.org/book", r.bookingUrl_);
  }

  // flex_trip_2: pickup = br_real_time (REAL_TIME)
  {
    ASSERT_TRUE(trip_2->from_.pickupBookingRule_.has_value());
    auto const& r = *trip_2->from_.pickupBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::REAL_TIME, r.bookingType_);
    EXPECT_FALSE(r.priorNoticeDurationMin_.has_value());
    EXPECT_FALSE(r.earliestBookingTime_.has_value());
    EXPECT_FALSE(r.latestBookingTime_.has_value());
    EXPECT_FALSE(r.message_.has_value());
    EXPECT_FALSE(r.phoneNumber_.has_value());
    EXPECT_EQ("https://example.org/book", r.bookingUrl_);
  }

  // flex_trip_2: drop off = br_same_day_unbounded (no max notice)
  {
    ASSERT_TRUE(trip_2->to_.dropOffBookingRule_.has_value());
    auto const& r = *trip_2->to_.dropOffBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::UP_TO_SAME_DAY, r.bookingType_);
    EXPECT_EQ(30, r.priorNoticeDurationMin_);
    EXPECT_FALSE(r.priorNoticeDurationMax_.has_value());
    EXPECT_FALSE(r.earliestBookingTime_.has_value());
    ASSERT_TRUE(r.latestBookingTime_.has_value());
    EXPECT_EQ(trip_2->startTime_.time_ - 30min, r.latestBookingTime_->time_);
    EXPECT_FALSE(r.message_.has_value());
  }

  // flex_trip_3: pickup = br_prior_service_day, days are counted only on
  // S_BOOK days (2019-04-26, 2019-04-29).
  {
    ASSERT_TRUE(trip_3->from_.pickupBookingRule_.has_value());
    auto const& r = *trip_3->from_.pickupBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::UP_TO_PRIOR_DAYS, r.bookingType_);
    ASSERT_TRUE(r.earliestBookingTime_.has_value());
    ASSERT_TRUE(r.latestBookingTime_.has_value());
    EXPECT_EQ(sys_days{2019_y / April / 26} + 7h,
              r.earliestBookingTime_->time_);
    EXPECT_EQ(sys_days{2019_y / April / 29} + 10h, r.latestBookingTime_->time_);
    EXPECT_FALSE(trip_3->to_.dropOffBookingRule_.has_value());
  }

  // JSON shape: camelCase keys and enum strings.
  auto const leg_json = json::serialize(json::value_from(*trip_1));
  EXPECT_NE(std::string::npos, leg_json.find(R"("pickupBookingRule":{)"));
  EXPECT_NE(std::string::npos, leg_json.find(R"("dropOffBookingRule":{)"));
  EXPECT_NE(std::string::npos,
            leg_json.find(R"("bookingType":"UP_TO_PRIOR_DAYS")"));
  EXPECT_NE(std::string::npos,
            leg_json.find(R"("latestBookingTime":"2019-04-30T15:00:00Z")"));
}

TEST(motis, flex_booking_rules_dst) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_booking_rules_dst", ec);

  auto const c = flex_config(kFlexDstGTFS, "2019-10-25", 6U);
  import(c, "test/data_flex_booking_rules_dst");
  auto d = data{"test/data_flex_booking_rules_dst", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  // Deadline: 17:00 local time on the day before the service day.
  // Expected values are UTC (CEST = UTC+2, CET = UTC+1).
  for (auto const& [travel, deadline] :
       {// deadline 2019-10-25 17:00 CEST
        std::pair{sys_days{2019_y / October / 26},
                  sys_days{2019_y / October / 25} + 15h},
        // deadline 2019-10-27 17:00 CET (the day DST ends)
        std::pair{sys_days{2019_y / October / 28},
                  sys_days{2019_y / October / 27} + 16h},
        // deadline 2019-10-26 17:00 CEST
        std::pair{sys_days{2019_y / October / 27},
                  sys_days{2019_y / October / 26} + 15h},
        // deadline 2019-10-28 17:00 CET
        std::pair{sys_days{2019_y / October / 29},
                  sys_days{2019_y / October / 28} + 16h}}) {
    auto const res =
        routing("?fromPlace=49.8760,8.6270&toPlace=49.8718,8.6305&time=" +
                date::format("%F", travel) +
                "T11:00Z&timetableView=false&transitModes=&directModes=FLEX");
    auto const* leg = find_flex_leg(res, "lg_1");
    ASSERT_NE(nullptr, leg) << travel;
    for (auto const& rule :
         {leg->from_.pickupBookingRule_, leg->to_.dropOffBookingRule_}) {
      ASSERT_TRUE(rule.has_value()) << travel;
      ASSERT_TRUE(rule->latestBookingTime_.has_value()) << travel;
      EXPECT_EQ(deadline, rule->latestBookingTime_->time_) << travel;
      EXPECT_FALSE(rule->earliestBookingTime_.has_value()) << travel;
    }
  }
}

TEST(motis, flex_without_booking_rules) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_no_booking_rules", ec);

  auto const c = flex_config(kFlexNoBookingRulesGTFS);
  import(c, "test/data_flex_no_booking_rules");
  auto d = data{"test/data_flex_no_booking_rules", c};
  EXPECT_TRUE(d.tt_->booking_rules_.empty());

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(kFlexQuery);

  auto n_flex_legs = 0U;
  for (auto const& it : res.direct_) {
    for (auto const& leg : it.legs_) {
      if (leg.mode_ != api::ModeEnum::FLEX) {
        continue;
      }
      ++n_flex_legs;
      EXPECT_FALSE(leg.from_.pickupBookingRule_.has_value());
      EXPECT_FALSE(leg.to_.dropOffBookingRule_.has_value());
      EXPECT_FALSE(leg.alerts_.has_value());
    }
  }
  EXPECT_EQ(1U, n_flex_legs);

  auto const res_json = json::serialize(json::value_from(res));
  EXPECT_EQ(std::string::npos, res_json.find("BookingRule"));
  EXPECT_EQ(std::string::npos, res_json.find("bookingType"));
}

TEST(motis, flex_alerts) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_alerts", ec);

  auto const c = flex_config(kFlexGTFS);
  import(c, "test/data_flex_alerts");
  auto d = data{"test/data_flex_alerts", c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto msg =
      test::to_feed_msg({test::alert{.header_ = "Booking required",
                                     .description_ = "Book via the Rufbus app.",
                                     .entities_ = {{.route_id_ = "FLEX_1"}}},
                         test::alert{.header_ = "No service today",
                                     .description_ = "Snow.",
                                     .entities_ = {{.route_id_ = "FLEX_1"}}}},
                        date::sys_days{2019_y / May / 1} + 6h);
  msg.mutable_entity(1)->mutable_alert()->set_severity_level(
      transit_realtime::Alert_SeverityLevel_WARNING);
  auto const stats = n::rt::gtfsrt_update_msg(*d.tt_, *d.rt_->rtt_,
                                              n::source_idx_t{0}, "test", msg);
  EXPECT_EQ(2U, stats.alert_total_resolve_success_);

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(kFlexQuery);

  auto n_flex_legs = 0U;
  for (auto const& it : res.direct_) {
    for (auto const& leg : it.legs_) {
      if (leg.mode_ != api::ModeEnum::FLEX) {
        continue;
      }
      ++n_flex_legs;
      if (leg.from_.flexId_ == "lg_1") {  // flex_trip_1 (FLEX_1)
        ASSERT_TRUE(leg.alerts_.has_value());
        ASSERT_EQ(2U, leg.alerts_->size());
        for (auto const& a : *leg.alerts_) {
          if (a.headerText_ == "Booking required") {
            EXPECT_EQ("Book via the Rufbus app.", a.descriptionText_);
            EXPECT_FALSE(a.severityLevel_.has_value());  // unknown -> omitted
          } else {
            EXPECT_EQ("No service today", a.headerText_);
            EXPECT_EQ(api::AlertSeverityLevelEnum::WARNING, a.severityLevel_);
          }
        }
      } else {  // flex_trip_2 (FLEX_2), flex_trip_3 (FLEX_3)
        EXPECT_FALSE(leg.alerts_.has_value());
      }
    }
  }
  EXPECT_EQ(3U, n_flex_legs);
}
