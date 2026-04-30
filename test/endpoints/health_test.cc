#include "gtest/gtest.h"

#include <filesystem>
#include <system_error>

#include "boost/asio/io_context.hpp"
#include "boost/beast/http/status.hpp"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/health.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"
#include "motis/metrics_registry.h"
#include "motis/rt_update.h"

using namespace motis;
using namespace testing;

TEST(motis, health_nofeeds) {
  auto const c =
      config{.timetable_ = {config::timetable{.datasets_ = {{"test", {}}}}}};

  auto const m = metrics_registry();
  auto const health = ep::health{.config_ = c, .metrics_ = &m};

  {
    auto const res = health("/api/v1/health");

    EXPECT_EQ(res.first, boost::beast::http::status::ok);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_FALSE(res.second.gbfs_.value());
  }
}

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
DA_Bus_1,DA Hbf,49.8724891,8.6281994

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
B1,Test,B1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
B1,S1,B1,Bus 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
B1,01:00:00,01:00:00,DA_Bus_1,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, health_feeds) {
  auto ec = std::error_code();
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .timetable_ = {config::timetable{
          .first_day_ = "2019-05-01",
          .num_days_ = 2,
          .datasets_ = {{"test",
                         {.path_ = kGTFS,
                          .clasz_bikes_allowed_ = {{{"LONGDISTANCE", false}}},
                          .rt_ = {{{.url_ = "https://example.test/rt"}}}}}}}},
      .gbfs_ = {{.feeds_ = {{"test", {.url_ = "https://example.test/gbfs"}}}}}};

  auto const m = metrics_registry{};
  auto const health = ep::health{.config_ = c, .metrics_ = &m};

  {
    // Feeds not consumed
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::bad_request);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_FALSE(res.second.gbfs_.value());
  }

  m.last_update_gbfs_.SetToCurrentTime();
  {
    // GBFS only consumed
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::bad_request);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_TRUE(res.second.gbfs_.value());
  }

  m.last_update_rt_.SetToCurrentTime();
  {
    // RT & GBFS consumed
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::ok);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_TRUE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_TRUE(res.second.gbfs_.value());
  }
}