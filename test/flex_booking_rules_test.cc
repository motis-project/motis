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

# location_group_stops.txt
location_group_id,stop_id
lg_1,A
lg_1,B
lg_2,A
lg_2,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_1,RUF,,Rufbus Musterland,715
FLEX_2,RUF,,Rufbus Musterland Express,715

# trips.txt
route_id,service_id,trip_id
FLEX_1,S1,flex_trip_1
FLEX_2,S1,flex_trip_2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max,prior_notice_last_day,prior_notice_last_time,prior_notice_start_day,prior_notice_start_time,message,pickup_message,drop_off_message,phone_number,info_url,booking_url
br_same_day,1,60,1440,,,,,Book at least 1 hour in advance.,,,+43 123 456 789,https://example.org/info,https://example.org/book
br_prior_day,2,,,1,17:00:00,7,08:00:00,Book until 17:00 the day before.,Pickup only after booking.,,+43 123 456 789,,
br_real_time,0,,,,,,,,,,,,https://example.org/book
br_same_day_unbounded,1,30,,,,,,,,,,,

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type,pickup_booking_rule_id,drop_off_booking_rule_id
flex_trip_1,1,lg_1,08:30:00,18:00:00,2,1,br_prior_day,
flex_trip_1,2,lg_1,08:30:00,18:00:00,1,2,,br_same_day
flex_trip_2,1,lg_2,08:30:00,18:00:00,2,1,br_real_time,
flex_trip_2,2,lg_2,08:30:00,18:00:00,1,2,,br_same_day_unbounded
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

config flex_config(std::string_view gtfs) {
  return config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
                .osm_ = {"test/resources/test_case.osm.pbf"},
                .timetable_ =
                    config::timetable{
                        .first_day_ = "2019-05-01",
                        .num_days_ = 2,
                        .datasets_ = {{"test", {.path_ = std::string{gtfs}}}}},
                .street_routing_ = true};
}

}  // namespace

TEST(motis, flex_booking_rules) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_booking_rules", ec);

  auto const c = flex_config(kFlexGTFS);
  import(c, "test/data_flex_booking_rules");
  auto d = data{"test/data_flex_booking_rules", c};

  // Import: booking rules reach the timetable.
  ASSERT_EQ(4U, d.tt_->booking_rules_.size());

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(kFlexQuery);

  auto flex_legs = std::vector<api::Leg const*>{};
  for (auto const& it : res.direct_) {
    for (auto const& leg : it.legs_) {
      if (leg.mode_ == api::ModeEnum::FLEX) {
        flex_legs.push_back(&leg);
      }
    }
  }
  ASSERT_EQ(2U, flex_legs.size());

  auto const* trip_1 = static_cast<api::Leg const*>(nullptr);
  auto const* trip_2 = static_cast<api::Leg const*>(nullptr);
  for (auto const* leg : flex_legs) {
    ASSERT_TRUE(leg->from_.pickupBookingRule_.has_value());
    if (leg->from_.pickupBookingRule_->bookingType_ ==
        api::BookingTypeEnum::UP_TO_PRIOR_DAYS) {
      trip_1 = leg;
    } else {
      trip_2 = leg;
    }
  }
  ASSERT_NE(nullptr, trip_1);
  ASSERT_NE(nullptr, trip_2);

  // flex_trip_1: pickup = br_prior_day (UP_TO_PRIOR_DAYS)
  {
    auto const& r = *trip_1->from_.pickupBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::UP_TO_PRIOR_DAYS, r.bookingType_);
    EXPECT_EQ(1, r.priorNoticeLastDay_);
    EXPECT_EQ("17:00:00", r.priorNoticeLastTime_);
    EXPECT_EQ(7, r.priorNoticeStartDay_);
    EXPECT_EQ("08:00:00", r.priorNoticeStartTime_);
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
    EXPECT_FALSE(r.priorNoticeLastDay_.has_value());
    EXPECT_FALSE(r.priorNoticeLastTime_.has_value());
    EXPECT_FALSE(r.priorNoticeStartDay_.has_value());
    EXPECT_FALSE(r.priorNoticeStartTime_.has_value());
    EXPECT_EQ("Book at least 1 hour in advance.", r.message_);
    EXPECT_FALSE(r.pickupMessage_.has_value());
    EXPECT_EQ("+43 123 456 789", r.phoneNumber_);
    EXPECT_EQ("https://example.org/info", r.infoUrl_);
    EXPECT_EQ("https://example.org/book", r.bookingUrl_);
  }

  // flex_trip_2: pickup = br_real_time (REAL_TIME)
  {
    auto const& r = *trip_2->from_.pickupBookingRule_;
    EXPECT_EQ(api::BookingTypeEnum::REAL_TIME, r.bookingType_);
    EXPECT_FALSE(r.priorNoticeDurationMin_.has_value());
    EXPECT_FALSE(r.priorNoticeLastDay_.has_value());
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
    EXPECT_FALSE(r.message_.has_value());
  }

  // JSON shape: camelCase keys and enum strings.
  auto const leg_json = json::serialize(json::value_from(*trip_1));
  EXPECT_NE(std::string::npos, leg_json.find(R"("pickupBookingRule":{)"));
  EXPECT_NE(std::string::npos, leg_json.find(R"("dropOffBookingRule":{)"));
  EXPECT_NE(std::string::npos,
            leg_json.find(R"("bookingType":"UP_TO_PRIOR_DAYS")"));
  EXPECT_NE(std::string::npos,
            leg_json.find(R"("priorNoticeLastTime":"17:00:00")"));
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
      ASSERT_TRUE(leg.from_.pickupBookingRule_.has_value());
      if (leg.from_.pickupBookingRule_->bookingType_ ==
          api::BookingTypeEnum::UP_TO_PRIOR_DAYS) {  // flex_trip_1 (FLEX_1)
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
      } else {  // flex_trip_2 (FLEX_2)
        EXPECT_FALSE(leg.alerts_.has_value());
      }
    }
  }
  EXPECT_EQ(2U, n_flex_legs);
}
