#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <system_error>

#include "boost/beast/http/status.hpp"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/health.h"

using namespace motis;
using namespace testing;

TEST(motis, health) {
  auto ec = std::error_code();
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.timetable_ = config::timetable{}, .gbfs_ = config::gbfs{}};

  auto d = data("test/data", c);

  auto const health = utl::init_from<ep::health>(d).value();

  auto const res = health("/api/v1/health");
  EXPECT_EQ(res.first, boost::beast::http::status::ok);
  EXPECT_TRUE(res.second.rt_.has_value());
  EXPECT_FALSE(res.second.rt_.value());
  EXPECT_TRUE(res.second.gbfs_.has_value());
  EXPECT_TRUE(res.second.gbfs_.value());
}