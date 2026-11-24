#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>

#include "utl/init_from.h"
#include "utl/to_vec.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/trips.h"
#include "motis/import.h"
#include "motis/railviz.h"

#include "./util.h"

using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;

// CEST (UTC+2): times below are given in UTC in the tests.
//   T:   A 08:00 -> B 08:10/08:20 (dwell) -> C 08:30
//   N:   C 23:50 -> D 00:20 (+1 day, crosses UTC midnight)
//   M:   A 20:00 -> B 08:00 (+1 day) -> C 08:00 (+2 days)
//   BUS: X 10:00 -> Y 10:10
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.00,8.00
B,B,50.10,8.00
C,C,50.20,8.00
D,D,50.30,8.00
X,X,50.00,8.01
Y,Y,50.00,8.02

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101
BUS,DB,BUS,,,3

# trips.txt
route_id,service_id,trip_id
ICE,SVC,T
ICE,SVC,N
ICE,SVC,M
BUS,SVC,BUS

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T,10:00:00,10:00:00,A,0
T,10:10:00,10:20:00,B,1
T,10:30:00,10:30:00,C,2
N,25:50:00,25:50:00,C,0
N,26:20:00,26:20:00,D,1
M,22:00:00,22:00:00,A,0
M,34:00:00,34:00:00,B,1
M,58:00:00,58:00:00,C,2
BUS,12:00:00,12:00:00,X,0
BUS,12:10:00,12:10:00,Y,1

# calendar_dates.txt
service_id,date,exception_type
SVC,20190501,1
)";

// NZST (UTC+12): Z runs 2019-04-30 20:00 -> 20:30 UTC.
constexpr auto const kNzGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
NZ,NZ,https://example.com,Pacific/Auckland

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
P,P,50.00,8.00
Q,Q,50.10,8.00

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,NZ,R,,,101

# trips.txt
route_id,service_id,trip_id
R,SVC,Z

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
Z,08:00:00,08:00:00,P,0
Z,08:30:00,08:30:00,Q,1

# calendar_dates.txt
service_id,date,exception_type
SVC,20190501,1
)";

// PDT (UTC-7): L runs 2019-05-02 03:00 -> 04:00 UTC.
constexpr auto const kLaGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
LA,LA,https://example.com,America/Los_Angeles

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
P,P,50.00,8.00
Q,Q,50.10,8.00

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,LA,R,,,101

# trips.txt
route_id,service_id,trip_id
R,SVC,L

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
L,20:00:00,20:00:00,P,0
L,21:00:00,21:00:00,Q,1

# calendar_dates.txt
service_id,date,exception_type
SVC,20190501,1
)";

// DST switch on 2019-03-31 02:00 CET -> 03:00 CEST.
constexpr auto const kDstGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.00,8.00
B,B,50.10,8.00

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id
ICE,DAILY,DAY
ICE,SWITCH,NIGHT

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
DAY,10:00:00,10:00:00,A,0
DAY,10:30:00,10:30:00,B,1
NIGHT,01:30:00,01:30:00,A,0
NIGHT,03:30:00,03:30:00,B,1

# calendar_dates.txt
service_id,date,exception_type
DAILY,20190330,1
DAILY,20190331,1
DAILY,20190401,1
SWITCH,20190331,1
)";

constexpr auto const kArea = "min=49.9,7.9&max=50.4,8.1";

namespace {

data import_gtfs(char const* dir,
                 date::sys_days const first_day,
                 std::map<std::string, std::string> const& feeds) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(dir, ec);

  auto c = config{.timetable_ =
                      config::timetable{.first_day_ = date::format("%F", first_day),
                                        .num_days_ = 4}};
  for (auto const& [tag, gtfs] : feeds) {
    c.timetable_->datasets_.emplace(tag, config::timetable::dataset{.path_ = gtfs});
  }
  import(c, dir);
  auto d = data{dir, c};
  d.init_rtt(first_day);
  d.load_railviz();
  return d;
}

std::string to_str(api::TripSegment const& s) {
  auto const time = [](openapi::date_time_t const t,
                       openapi::date_time_t const scheduled) {
    return date::format("%d %R", *t) +
           (t == scheduled ? "" : date::format(" (%R)", *scheduled));
  };
  return s.trips_.at(0).tripId_ + " " + s.from_.name_ + " " +
         time(s.departure_, s.scheduledDeparture_) + " -> " + s.to_.name_ +
         " " + time(s.arrival_, s.scheduledArrival_) +
         (s.realTime_ ? " rt" : "");
}

std::vector<std::string> segments(data& d,
                                  std::string_view start,
                                  std::string_view end,
                                  std::string_view area = kArea,
                                  int const zoom = 18) {
  auto const trips = utl::init_from<ep::trips>(d).value();
  auto res = utl::to_vec(
      trips(fmt::format("/api/v5/map/trips?zoom={}&{}&precision=5"
                        "&startTime={}&endTime={}",
                        zoom, area, start, end)),
      to_str);
  std::ranges::sort(res);
  return res;
}

using strings = std::vector<std::string>;

}  // namespace

TEST(motis, railviz) {
  auto d = import_gtfs("test/data_railviz", 2019_y / May / 1, {{"test", kGTFS}});

  // Moving between stops.
  EXPECT_EQ(strings{"20190501_10:00_test_T A 01 08:00 -> B 01 08:10"},
            segments(d, "2019-05-01T08:05:00Z", "2019-05-01T08:06:00Z"));

  // Dwelling at B: segment B -> C is returned.
  EXPECT_EQ(strings{"20190501_10:00_test_T B 01 08:20 -> C 01 08:30"},
            segments(d, "2019-05-01T08:12:00Z", "2019-05-01T08:15:00Z"));

  // Before the trip.
  EXPECT_EQ(strings{},
            segments(d, "2019-05-01T07:50:00Z", "2019-05-01T07:55:00Z"));

  // Outside of the area.
  EXPECT_EQ(strings{}, segments(d, "2019-05-01T08:05:00Z",
                                "2019-05-01T08:06:00Z", "min=40,0&max=41,1"));

  // Not operating on that day (only M is still running).
  EXPECT_EQ(strings{"20190501_22:00_test_M B 02 08:00 -> C 03 08:00"},
            segments(d, "2019-05-02T08:05:00Z", "2019-05-02T08:06:00Z"));

  // Segment crossing UTC midnight, queried after midnight.
  EXPECT_EQ((strings{"20190501_22:00_test_M A 01 20:00 -> B 02 08:00",
                     "20190501_25:50_test_N C 01 23:50 -> D 02 00:20"}),
            segments(d, "2019-05-02T00:10:00Z", "2019-05-02T00:15:00Z"));

  // Multi-day trip: segment on the day after the service day + 1.
  EXPECT_EQ(strings{"20190501_22:00_test_M B 02 08:00 -> C 03 08:00"},
            segments(d, "2019-05-03T07:00:00Z", "2019-05-03T07:01:00Z"));

  // Zoom: bus only shown when zoomed in, ICE also when zoomed out.
  EXPECT_EQ(strings{"20190501_12:00_test_BUS X 01 10:00 -> Y 01 10:10"},
            segments(d, "2019-05-01T10:02:00Z", "2019-05-01T10:03:00Z"));
  EXPECT_EQ(strings{}, segments(d, "2019-05-01T10:02:00Z",
                                "2019-05-01T10:03:00Z", kArea, 8));
  EXPECT_EQ(strings{"20190501_10:00_test_T A 01 08:00 -> B 01 08:10"},
            segments(d, "2019-05-01T08:05:00Z", "2019-05-01T08:06:00Z", kArea,
                     4));

  // Real-time: delay.
  {
    auto const stats = gtfsrt_update(
        d, to_feed_msg(
               {trip_update{.trip_ = {.trip_id_ = "T", .date_ = {"20190501"}},
                            .stop_updates_ = {{.stop_id_ = "A",
                                               .delay_minutes_ = 1}}}},
               date::sys_days{2019_y / May / 1} + 8h));
    EXPECT_EQ(1U, stats.total_entities_success_);
    d.load_railviz();

    EXPECT_EQ(
        strings{"20190501_10:00_test_T A 01 08:01 (08:00) -> B 01 08:11 "
                "(08:10) rt"},
        segments(d, "2019-05-01T08:05:00Z", "2019-05-01T08:06:00Z"));
    EXPECT_EQ(
        strings{"20190501_10:00_test_T B 01 08:21 (08:20) -> C 01 08:31 "
                "(08:30) rt"},
        segments(d, "2019-05-01T08:12:00Z", "2019-05-01T08:15:00Z"));
  }

  // Real-time: skipped stop -> one segment across it.
  {
    d.init_rtt(date::sys_days{2019_y / May / 1});
    auto const stats = gtfsrt_update(
        d, to_feed_msg(
               {trip_update{.trip_ = {.trip_id_ = "T", .date_ = {"20190501"}},
                            .stop_updates_ = {{.stop_id_ = "B",
                                               .skip_ = true}}}},
               date::sys_days{2019_y / May / 1} + 8h));
    EXPECT_EQ(1U, stats.total_entities_success_);
    d.load_railviz();

    EXPECT_EQ(strings{"20190501_10:00_test_T A 01 08:00 -> C 01 08:30"},
              segments(d, "2019-05-01T08:05:00Z", "2019-05-01T08:06:00Z"));
  }

  // Real-time: cancelled trip.
  {
    d.init_rtt(date::sys_days{2019_y / May / 1});
    auto const stats = gtfsrt_update(
        d,
        to_feed_msg({trip_update{
                        .trip_ = {.trip_id_ = "N", .date_ = {"20190501"}},
                        .cancelled_ = true}},
                    date::sys_days{2019_y / May / 1} + 8h));
    EXPECT_EQ(1U, stats.total_entities_success_);
    d.load_railviz();

    EXPECT_EQ(strings{"20190501_22:00_test_M A 01 20:00 -> B 02 08:00"},
              segments(d, "2019-05-02T00:10:00Z", "2019-05-02T00:15:00Z"));
  }
}

TEST(motis, railviz_timezones) {
  auto d = import_gtfs("test/data_railviz_timezones", 2019_y / May / 1,
                       {{"nz", kNzGTFS}, {"la", kLaGTFS}});

  // Local service day 2019-05-01 starts on 2019-04-30 in UTC.
  EXPECT_EQ(strings{"20190501_08:00_nz_Z P 30 20:00 -> Q 30 20:30"},
            segments(d, "2019-04-30T20:10:00Z", "2019-04-30T20:11:00Z"));

  // Local service day 2019-05-01 ends on 2019-05-02 in UTC.
  EXPECT_EQ(strings{"20190501_20:00_la_L P 02 03:00 -> Q 02 04:00"},
            segments(d, "2019-05-02T03:10:00Z", "2019-05-02T03:11:00Z"));
}

TEST(motis, railviz_dst) {
  auto d = import_gtfs("test/data_railviz_dst", 2019_y / March / 30,
                       {{"test", kDstGTFS}});

  // 10:00 local: 09:00 UTC (CET) before, 08:00 UTC (CEST) after the switch.
  EXPECT_EQ(strings{"20190330_10:00_test_DAY A 30 09:00 -> B 30 09:30"},
            segments(d, "2019-03-30T09:10:00Z", "2019-03-30T09:11:00Z"));
  EXPECT_EQ(strings{},
            segments(d, "2019-03-31T09:10:00Z", "2019-03-31T09:11:00Z"));
  EXPECT_EQ(strings{"20190331_10:00_test_DAY A 31 08:00 -> B 31 08:30"},
            segments(d, "2019-03-31T08:10:00Z", "2019-03-31T08:11:00Z"));
  EXPECT_EQ(strings{"20190401_10:00_test_DAY A 01 08:00 -> B 01 08:30"},
            segments(d, "2019-04-01T08:10:00Z", "2019-04-01T08:11:00Z"));

  // Night of the switch: GTFS times are relative to noon - 12h (23:00 CET).
  EXPECT_EQ(strings{"20190331_01:30_test_NIGHT A 30 23:30 -> B 31 01:30"},
            segments(d, "2019-03-30T23:40:00Z", "2019-03-30T23:41:00Z"));
  EXPECT_EQ(strings{"20190331_01:30_test_NIGHT A 30 23:30 -> B 31 01:30"},
            segments(d, "2019-03-31T01:00:00Z", "2019-03-31T01:01:00Z"));
}
