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

std::map<std::string, double> scrape(data& d, metrics_registry& registry) {
  auto const reply = ep::metrics{.tt_ = d.tt_.get(),
                                 .tags_ = d.tags_.get(),
                                 .rt_ = d.rt_,
                                 .metrics_ = &registry}(
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
  // Service day such that all stop times (relative to now) are within 0-30h.
  auto const now = std::chrono::floor<std::chrono::minutes>(
      std::chrono::system_clock::now());
  auto const day = std::chrono::floor<date::days>(now - 2h);
  auto const t = [&](std::chrono::minutes const rel) {
    auto const m = (now + rel - day) / 1min;
    return fmt::format("{:02}:{:02}:00", m / 60, m % 60);
  };

  auto const gtfs = fmt::format(R"(
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

# trips.txt
route_id,service_id,trip_id
R,S,STATIC
R,S,DELAYED
R,S,EARLY
R,S,FUTURE
B,S,CANCELLED

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

# calendar_dates.txt
service_id,date,exception_type
S,{4:%Y%m%d},1
)",
                                t(-60min), t(60min), t(120min), t(180min), day);

  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_metrics", ec);
  auto const c = config{
      .timetable_ = config::timetable{.first_day_ = date::format(
                                          "%F", day - date::days{1}),
                                      .num_days_ = 3,
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
                   .added_ = true}},
      now);
  // to_feed_msg() omits the start date for cancelled trips
  msg.mutable_entity(3)->mutable_trip_update()->mutable_trip()->set_start_date(
      service_date);
  auto const stats = n::rt::gtfsrt_update_msg(*d.tt_, *d.rt_->rtt_,
                                              n::source_idx_t{0}, "test", msg);
  EXPECT_EQ(5U, stats.total_entities_success_);

  auto registry = metrics_registry{};
  auto m = scrape(d, registry);
  using labels = std::map<std::string, std::string>;
  auto const rail = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "DB", "REGIONAL_RAIL", l));
  };
  auto const bus = [&](std::string_view name, labels const& l = {}) {
    return m.at(series(name, "BUS", "BUS", l));
  };
  auto const realtime = "current_trips_running_scheduled_with_realtime_count";
  auto const delay = std::string{"current_trips_running_delay_minutes"};
  auto const headway = "current_trips_running_delay_headway_count";

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

  // Series without RT data anymore are reset.
  d.init_rtt();
  m = scrape(d, registry);
  EXPECT_EQ(3, rail("current_trips_running_scheduled_count"));
  EXPECT_EQ(0, rail(realtime, {{"cancelled", "false"}}));
  EXPECT_EQ(0, rail(delay + "_count"));
  EXPECT_EQ(0, rail(headway, {{"multiple", "true"}}));
  EXPECT_EQ(1, bus("current_trips_running_scheduled_count"));
  EXPECT_EQ(0, bus(realtime, {{"cancelled", "true"}}));
  EXPECT_EQ(0, bus("current_trips_running_added_count"));
}
