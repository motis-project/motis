#include "gtest/gtest.h"

#include "boost/date_time/posix_time/posix_time.hpp"

#include "motis/bikesharing/terminal.h"

namespace motis::bikesharing {

TEST(bikesharing_terminal, timestamp_to_bucket) {
  // sunday = 0
  size_t bucket = 3 * kHoursPerDay + 10;

  std::time_t t0 = 1453280400;  // Wed, 20 Jan 2016 10:00:0 GMT+0100
  EXPECT_EQ(bucket, timestamp_to_bucket(t0));

  std::time_t t1 = 1453283999;  // Wed, 20 Jan 2016 10:59:59 GMT+0100
  EXPECT_EQ(bucket, timestamp_to_bucket(t1));

  std::time_t t2 = 1453886836;  // Wed, 27 Jan 2016 10:27:16 GMT+0100
  EXPECT_EQ(bucket, timestamp_to_bucket(t2));

  std::time_t t3 = 1453282037;  // Wed, 13 Apr 2016 10:30:00 GMT+0200
  EXPECT_EQ(bucket, timestamp_to_bucket(t3));
}

}  // namespace motis::bikesharing
