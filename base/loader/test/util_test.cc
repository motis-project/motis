#include <cinttypes>
#include <cstring>
#include <functional>

#include "gtest/gtest.h"

#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "motis/core/common/date_time_util.h"
#include "motis/schedule/bitfield.h"
#include "motis/loader/util.h"

using namespace utl;

namespace motis::loader::hrd {

TEST(loader_util, bitset_to_string_and_back) {
  std::string bit_string = "0101010100101010";
  std::bitset<16> before(bit_string);

  ASSERT_TRUE(deserialize_bitset<16>(serialize_bitset<16>(before).c_str()) ==
              before);
}

TEST(loader_util, raw_to_int) {
  ASSERT_TRUE(raw_to_int<uint16_t>("ab") == 97 + (98U << 8U));
}

TEST(loader_util, hhmm_to_int_1) {
  ASSERT_TRUE(hhmm_to_min(parse<int>("0130")) == 90);
}

TEST(loader_util, hhmm_to_int_2) {
  ASSERT_TRUE(hhmm_to_min(parse<int>("")) == 0);
}

TEST(loader_util, hhmm_to_int_3) {
  ASSERT_TRUE(hhmm_to_min(parse<int>("2501")) == 1501);
}

TEST(loader_util, find_nth_1st) {
  auto ints = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2};
  auto it =
      find_nth(begin(ints), end(ints), 1, [](auto&& num) { return num == 2; });
  ASSERT_TRUE(it != end(ints));
  ASSERT_TRUE(std::distance(begin(ints), it) == 1);
}

TEST(loader_util, find_nth_2nd) {
  auto ints = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2};
  auto it =
      find_nth(begin(ints), end(ints), 2, [](auto&& num) { return num == 2; });
  ASSERT_TRUE(it != end(ints));
  ASSERT_TRUE(std::distance(begin(ints), it) == 7);
}

TEST(loader_util, find_nth_not_found_contained) {
  auto ints = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2};
  auto it =
      find_nth(begin(ints), end(ints), 4, [](auto&& num) { return num == 2; });
  ASSERT_TRUE(it == end(ints));
}

TEST(loader_util, find_nth_not_found_not_contained) {
  auto ints = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2};
  auto it =
      find_nth(begin(ints), end(ints), 1, [](auto&& num) { return num == 7; });
  ASSERT_TRUE(it == end(ints));
}

TEST(loader_util, find_nth_not_found_empty_vec) {
  auto ints = std::vector<int>();
  auto it =
      find_nth(begin(ints), end(ints), 1, [](auto&& num) { return num == 7; });
  ASSERT_TRUE(it == end(ints));
}

}  // namespace motis::loader::hrd
