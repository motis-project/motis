#include "gtest/gtest.h"

#include "motis/path/polyline_format.h"

namespace mp = motis::path;

// the official example from :
// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
TEST(path_polyline_format_test, google_coord) {
  mp::polyline_encoder enc;
  enc.push_difference(-179.9832104 * mp::polyline_encoder<>::kPrecision);
  EXPECT_EQ("`~oia@", enc.buf_);

  auto const line = mp::decode_polyline("`~oia@");
  ASSERT_EQ(1, line.size());
  EXPECT_EQ(-179.98321, line[0].lat_);
  EXPECT_EQ(0, line[0].lng_);
}

TEST(path_polyline_format_test, google_polyline) {
  geo::polyline original{{38.5, -120.2}, {40.7, -120.95}, {43.252, -126.453}};

  auto const encoded = mp::encode_polyline(original);
  EXPECT_EQ("_p~iF~ps|U_ulLnnqC_mqNvxq`@", encoded);

  auto const decoded = mp::decode_polyline(encoded);
  EXPECT_EQ(original, decoded);
}
