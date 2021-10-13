#include "gtest/gtest.h"

#include "utl/parser/cstr.h"

#include "motis/loader/hrd/parser/bitfields_parser.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;

namespace motis::loader::hrd {

TEST(loader_hrd_bitfields, to_bitset_invalid_period_no_1) {
  bool catched = false;
  try {
    hex_str_to_bitset("0", "file.101", 1);
  } catch (parser_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_invalid_period_one_1) {
  bool catched = false;
  try {
    hex_str_to_bitset("1", "file.101", 1);
  } catch (parser_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_invalid_period_two_1) {
  bool catched = false;
  try {
    hex_str_to_bitset("3", "file.101", 1);
  } catch (parser_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_invalid_period_three_1) {
  bool catched = false;
  try {
    hex_str_to_bitset("7", "file.101", 1);
  } catch (parser_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_invalid_period_four_1) {
  bool catched = false;
  try {
    hex_str_to_bitset("F", "file.101", 1);
  } catch (parser_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_valid_period_1) {
  // 0x0653 = 0000 0110 0101 0011
  ASSERT_TRUE(std::bitset<MAX_DAYS>("0010100") ==
              hex_str_to_bitset("0653", "file.101", 1));
}

TEST(loader_hrd_bitfields, hex_str_to_bitset_valid_period_2) {
  // 0xC218 = 1100 0010 0001 1000
  ASSERT_TRUE(std::bitset<MAX_DAYS>("000010000") ==
              hex_str_to_bitset("C218", "file.101", 1));
}

}  // namespace motis::loader::hrd
