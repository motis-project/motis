#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <system_error>

#include "boost/asio/io_context.hpp"
#include "boost/beast/http/status.hpp"

#include "motis/rt_update.h"
#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/health.h"
#include "motis/gbfs/update.h"

using namespace motis;
using namespace testing;

TEST(motis, health_nofeeds) {
  auto const c = config{};

  auto d = data("", c);

  auto const health = utl::init_from<ep::health>(d).value();

  // No feeds
  {
    auto const res = health("/api/v1/health");

    EXPECT_EQ(res.first, boost::beast::http::status::ok);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_FALSE(res.second.gbfs_.value());
  }
}

TEST(motis, health_feeds) {
  auto const c =
      config{.timetable_ = {config::timetable{
                 .datasets_ = {{"test",
                                {.path_ = "",
                                 .clasz_bikes_allowed_ = {{{"", false}}},
                                 .rt_ = {{{.url_ = ""}}}}}}}},
             .gbfs_ = {{.feeds_ = {{"test", {.url_ = ""}}}}}};

  auto d = data("", c);

  auto const health = utl::init_from<ep::health>(d).value();

  // Feeds not consumed
  {
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::bad_request);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_FALSE(res.second.gbfs_.value());
  }

  // GBFS consumed
  {
    auto ioc = boost::asio::io_context{};
    gbfs::run_gbfs_update(ioc, c, *d.w_, *d.l_, d.gbfs_, d.metrics_.get());
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::ok);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_FALSE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_TRUE(res.second.gbfs_.value());
  }

  // RT & GBFS consumed
  {
    auto ioc = boost::asio::io_context{};
    run_rt_update(ioc, c, d);
    auto const res = health("api/v1/health");
    EXPECT_EQ(res.first, boost::beast::http::status::ok);
    EXPECT_TRUE(res.second.rt_.has_value());
    EXPECT_TRUE(res.second.rt_.value());
    EXPECT_TRUE(res.second.gbfs_.has_value());
    EXPECT_TRUE(res.second.gbfs_.value());
  }
}