#include "gtest/gtest.h"

#include <chrono>
#include <map>
#include <sstream>
#include <string>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "fmt/format.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/metrics.h"
#include "motis/import.h"
#include "motis/metrics_registry.h"
#include "motis/tag_lookup.h"

#include "../util.h"

using namespace motis;
using namespace motis::test;
using namespace std::chrono_literals;
namespace n = nigiri;

namespace {

std::map<std::string, double> scrape(
    data& d,
    metrics_registry& registry,
    std::chrono::system_clock::time_point const now) {
  auto const reply = ep::metrics{.tt_ = d.tt_.get(),
                                 .tags_ = d.tags_.get(),
                                 .rt_ = d.rt_,
                                 .metrics_ = &registry,
                                 .now_ = now}(
      net::route_request{net::web_server::http_req_t{
          boost::beast::http::verb::get, "/metrics", 11}},
      false);
  auto values = std::map<std::string, double>{};
  auto body =
      std::istringstream{std::get<net::web_server::string_res_t>(reply).body()};
  auto line = std::string{};
  while (std::getline(body, line)) {
    if (!line.starts_with("#")) {
      auto const split = line.rfind(' ');
      values[line.substr(0, split)] = std::stod(line.substr(split + 1U));
    }
  }
  return values;
}

std::string series(std::string_view name,
                   std::string_view agency,
                   std::string_view mode,
                   std::map<std::string, std::string> labels) {
  auto const le = labels.extract("le");
  labels.insert({{"agency_id", std::string{agency}},
                 {"agency_name", std::string{agency}},
                 {"mode", std::string{mode}},
                 {"tag", "test"}});
  auto s = fmt::format("{}{{", name);
  for (auto const& [k, v] : labels) {
    s += fmt::format(R"({}="{}",)", k, v);
  }
  if (!le.empty()) {
    s += fmt::format(R"(le="{}",)", le.mapped());
  }
  s.back() = '}';
  return s;
}

}  // namespace

TEST(motis, metrics_running_trips) {
  // Fixed time of day: trips of a route that cross UTC midnight at different
  // stops are split into separate routes (nigiri::loader::get_index).
  auto const day =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const now = std::chrono::sys_time<std::chrono::minutes>{day + 12h};
  auto const long_day = day - date::days{4};
  auto const long_next_day = day - date::days{3};
  auto const t_on = [&](date::sys_days const service_day,
                        std::chrono::minutes const rel) {
    auto const m = (now + rel - service_day) / 1min;
    return fmt::format("{:02}:{:02}:00", m / 60, m % 60);
  };
  auto const t = [&](std::chrono::minutes const rel) { return t_on(day, rel); };

  auto const gtfs = fmt::format(
      R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Etc/UTC
BUS,BUS,https://bus.de,Etc/UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,49.0,8.0
B,B,49.1,8.0
C,C,49.2,8.0

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,DB,R,,,2
B,BUS,B,,,3
T,DB,T,,,0
F,DB,F,,,4

# trips.txt
route_id,service_id,trip_id
R,S,STATIC
R,S,DELAYED
R,S,EARLY
R,S,FUTURE
B,S,CANCELLED
T,S,T1
T,S,T2
T,S,DAY
T,S,DAY_NEXT
F,L,LONG
F,L2,LONG_NEXT

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
STATIC,{0},{0},A,1
STATIC,{1},{1},B,2
DELAYED,{0},{0},A,1
DELAYED,{1},{1},B,2
EARLY,{0},{0},A,1
EARLY,{1},{1},B,2
FUTURE,{2},{2},A,1
FUTURE,{3},{3},B,2
CANCELLED,{0},{0},A,1
CANCELLED,{1},{1},C,2
T1,{5},{5},B,1
T1,{6},{6},C,2
T2,{7},{7},B,1
T2,{8},{8},C,2
DAY,{9},{9},C,1
DAY,{10},{10},B,2
DAY_NEXT,{11},{11},C,1
DAY_NEXT,{12},{12},B,2
LONG,{13},{13},A,1
LONG,{14},{14},B,2
LONG,{15},{15},C,3
LONG_NEXT,{16},{16},A,1
LONG_NEXT,{17},{17},B,2
LONG_NEXT,{18},{18},C,3

# calendar_dates.txt
service_id,date,exception_type
S,{4:%Y%m%d},1
L,{19:%Y%m%d},1
L2,{20:%Y%m%d},1
)",
      t(-60min), t(60min), t(120min), t(180min), day, t(-100min), t(100min),
      t(-40min), t(160min), t(1430min), t(1450min), t(1437min), t(1457min),
      t_on(long_day, -4 * 1440min), t_on(long_day, 710min),
      t_on(long_day, 718min), t_on(long_next_day, -3 * 1440min - 710min),
      t_on(long_next_day, 725min), t_on(long_next_day, 740min), long_day,
      long_next_day);

  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_metrics", ec);
  auto const c = config{
      .timetable_ = config::timetable{.first_day_ = date::format(
                                          "%F", long_day - date::days{1}),
                                      .num_days_ = 8,
                                      .datasets_ = {{"test", {.path_ = gtfs}}}},
      .street_routing_ = false};
  import(c, "test/data_metrics");
  auto d = data{"test/data_metrics", c};
  d.init_rtt();

  auto const service_date = date::format("%Y%m%d", day);
  auto msg = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "DELAYED", .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "A", .delay_minutes_ = 7}}},
       trip_update{.trip_ = {.trip_id_ = "EARLY", .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "A", .delay_minutes_ = -10}}},
       trip_update{
           .trip_ = {.trip_id_ = "FUTURE", .date_ = service_date},
           .stop_updates_ = {{.stop_id_ = "A", .delay_minutes_ = -130}}},
       trip_update{.trip_ = {.trip_id_ = "CANCELLED"}, .cancelled_ = true},
       trip_update{.trip_ = {.trip_id_ = "ADDED", .route_id_ = "B"},
                   .stop_updates_ = {{.stop_id_ = "A", .time_ = now - 10min},
                                     {.stop_id_ = "C", .time_ = now + 10min}},
                   .added_ = true},
       trip_update{.trip_ = {.trip_id_ = "T1", .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "B", .delay_minutes_ = 80}}},
       trip_update{
           .trip_ = {.trip_id_ = "DAY", .date_ = service_date},
           .stop_updates_ = {{.stop_id_ = "C", .delay_minutes_ = -1440}}},
       trip_update{.trip_ = {.trip_id_ = "LONG",
                             .date_ = date::format("%Y%m%d", long_day)},
                   .stop_updates_ = {{.stop_id_ = "B",
                                      .ev_type_ = n::event_type::kArr,
                                      .delay_minutes_ = 25}}},
       trip_update{.trip_ = {.trip_id_ = "LONG_NEXT",
                             .date_ = date::format("%Y%m%d", long_next_day)},
                   .stop_updates_ = {{.stop_id_ = "B",
                                      .ev_type_ = n::event_type::kArr,
                                      .delay_minutes_ = -15}}}},
      now);
  // to_feed_msg() omits the start date for cancelled trips
  msg.mutable_entity(3)->mutable_trip_update()->mutable_trip()->set_start_date(
      service_date);
  auto const stats = n::rt::gtfsrt_update_msg(*d.tt_, *d.rt_->rtt_,
                                              n::source_idx_t{0}, "test", msg);
  EXPECT_EQ(9U, stats.total_entities_success_);

  auto registry = metrics_registry{};
  auto m = scrape(d, registry, now);
  using labels = std::map<std::string, std::string>;
  auto const rail = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "DB", "REGIONAL_RAIL", l));
  };
  auto const bus = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "BUS", "BUS", l));
  };
  auto const tram = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "DB", "TRAM", l));
  };
  auto const ferry = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "DB", "FERRY", l));
  };
  auto const realtime = "current_trips_running_scheduled_with_realtime_count";
  auto const delay = std::string{"current_trips_running_delay_minutes"};
  auto const headway = "current_trips_running_delay_headway_count";
  auto const rt_events = "total_rt_events_count";
  auto const predicted = labels{{"timestamp_reason", "PREDICTED"}};
  auto const propagated = labels{{"timestamp_reason", "PROPAGATED"}};

  EXPECT_EQ(4, rail("current_trips_running_scheduled_count"));
  EXPECT_EQ(3, rail(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(0, rail(realtime, {{"cancelled", "true"}}));
  EXPECT_EQ(3, rail(delay + "_count"));
  EXPECT_EQ(-133, rail(delay + "_sum"));
  EXPECT_EQ(2, rail(delay + "_bucket", {{"le", "-10"}}));
  EXPECT_EQ(2, rail(delay + "_bucket", {{"le", "-4"}}));
  EXPECT_EQ(2, rail(delay + "_bucket", {{"le", "5"}}));
  EXPECT_EQ(3, rail(delay + "_bucket", {{"le", "10"}}));
  // FUTURE: -130 min at 120 min headway
  EXPECT_EQ(1, rail(headway, {{"multiple", "true"}}));
  EXPECT_EQ(0, rail(headway, {{"multiple", "false"}}));

  EXPECT_EQ(1, bus("current_trips_running_scheduled_count"));
  EXPECT_EQ(0, bus(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(1, bus(realtime, {{"cancelled", "true"}}));
  EXPECT_EQ(1, bus("current_trips_running_added_count"));
  EXPECT_FALSE(m.contains(series(delay + "_count", "BUS", "BUS", {})));

  EXPECT_EQ(3, tram("current_trips_running_scheduled_count"));
  EXPECT_EQ(2, tram(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(2, tram(delay + "_count"));
  EXPECT_EQ(-1360, tram(delay + "_sum"));
  // DAY: -1440 min at 7 min headway
  EXPECT_EQ(1, tram(headway, {{"multiple", "true"}}));
  // T1: +80 min at 60 min headway
  EXPECT_EQ(1, tram(headway, {{"multiple", "false"}}));

  // LONG (departed 4 days ago) passes B at 23:50, LONG_NEXT (departed 3 days
  // ago) at 00:05 on the next day.
  EXPECT_EQ(2, ferry("current_trips_running_scheduled_count"));
  EXPECT_EQ(2, ferry(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(10, ferry(delay + "_sum"));
  // LONG_NEXT: -15 min at 15 min headway
  EXPECT_EQ(1, ferry(headway, {{"multiple", "true"}}));
  // LONG: +25 min at 15 min headway
  EXPECT_EQ(1, ferry(headway, {{"multiple", "false"}}));

  EXPECT_EQ(3, rail(rt_events, predicted));
  EXPECT_EQ(3, rail(rt_events, propagated));
  EXPECT_EQ(0, rail(rt_events, {{"timestamp_reason", "OBSERVED"}}));
  EXPECT_EQ(2, bus(rt_events, predicted));
  EXPECT_EQ(0, bus(rt_events, propagated));
  EXPECT_EQ(2, tram(rt_events, predicted));
  EXPECT_EQ(2, tram(rt_events, propagated));
  EXPECT_EQ(8, m.at("total_trips_with_realtime_count"));

  // UTC days of the first (LONG) and last (DAY_NEXT) departure
  auto const day_start = [](date::sys_days const x) {
    return static_cast<double>(
        std::chrono::sys_seconds{x}.time_since_epoch().count());
  };
  auto const tag = R"({tag="test"})";
  EXPECT_EQ(
      day_start(long_day),
      m.at(fmt::format("nigiri_timetable_first_day_timestamp_seconds{}", tag)));
  EXPECT_EQ(
      day_start(day + date::days{1}) + 86399,
      m.at(fmt::format("nigiri_timetable_last_day_timestamp_seconds{}", tag)));
  EXPECT_EQ(3, m.at(fmt::format("nigiri_timetable_locations_count{}", tag)));
  EXPECT_EQ(11, m.at(fmt::format("nigiri_timetable_trips_count{}", tag)));
  EXPECT_EQ(
      11, m.at(fmt::format("nigiri_timetable_transports_x_days_count{}", tag)));

  // Cancel DELAYED, which already has an RT transport.
  auto cancel = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "DELAYED"}, .cancelled_ = true}}, now);
  cancel.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_trip()
      ->set_start_date(service_date);
  EXPECT_EQ(1U, n::rt::gtfsrt_update_msg(*d.tt_, *d.rt_->rtt_,
                                         n::source_idx_t{0}, "test", cancel)
                    .total_entities_success_);
  m = scrape(d, registry, now);
  EXPECT_EQ(4, rail("current_trips_running_scheduled_count"));
  EXPECT_EQ(2, rail(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(1, rail(realtime, {{"cancelled", "true"}}));
  EXPECT_EQ(2, rail(delay + "_count"));

  // Series without RT data anymore are reset.
  d.init_rtt();
  m = scrape(d, registry, now);
  EXPECT_EQ(3, rail("current_trips_running_scheduled_count"));
  EXPECT_EQ(0, rail(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(0, rail(delay + "_count"));
  EXPECT_EQ(0, rail(headway, {{"multiple", "true"}}));
  EXPECT_EQ(1, bus("current_trips_running_scheduled_count"));
  EXPECT_EQ(0, bus(realtime, {{"cancelled", "true"}}));
  EXPECT_EQ(0, bus("current_trips_running_added_count"));
  EXPECT_EQ(0, rail(rt_events, predicted));
  EXPECT_EQ(0, m.at("total_trips_with_realtime_count"));
}
