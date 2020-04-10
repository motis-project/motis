#include "gtest/gtest.h"

#include "motis/ris/zip_reader.h"

#include "./sha1.hpp"

namespace motis::ris {

TEST(ris_zip_reader, read_zip) {
  zip_reader r{"modules/ris/test_resources/test.zip"};
  std::optional<std::string_view> file_content;

  // f9735b6d730a91f9a77af2ac17fcb833  test.200.raw
  ASSERT_TRUE(file_content = r.read());
  EXPECT_EQ(std::string("dbaf3e2fc999af78c9e1af23b8a35aad1e8aec3d"),
            sha1::to_hex_str(sha1::sha1(*file_content)));

  // d5715e91ac27550e94728d3704bcac52  test.5000.raw
  ASSERT_TRUE(file_content = r.read());
  EXPECT_EQ(std::string("36a939a26f52eb9974f91a92ceabf47198ad7f6d"),
            sha1::to_hex_str(sha1::sha1(*file_content)));
}

}  // namespace motis::ris
