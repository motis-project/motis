#include <cinttypes>
#include <cstring>

#include "gtest/gtest.h"

#include "motis/loader/hrd/parser/attributes_parser.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;

namespace motis::loader::hrd {

TEST(loader_hrd_attributes, parse_line) {
  for (auto const& c : configs) {
    loaded_file f = {c.files(ATTRIBUTES),
                     ",  0 260 10 Bus mit Fahrradanhänger#"};
    auto attributes = parse_attributes(f, c);
    ASSERT_TRUE(attributes.size() == 1);

    auto it = attributes.find(raw_to_int<uint16_t>(", "));
    ASSERT_TRUE(it != end(attributes));
    ASSERT_TRUE(it->second == "Bus mit Fahrradanhänger");
  }
}

TEST(loader_hrd_attributes, parse_and_ignore_line) {
  for (auto const& c : configs) {
    loaded_file f = {c.files(ATTRIBUTES),
                     "ZZ 0 060 10 zusätzlicher Zug#\n# ,  ,  ,"};
    auto attributes = parse_attributes(f, c);
    ASSERT_TRUE(attributes.size() == 1);

    auto it = attributes.find(raw_to_int<uint16_t>("ZZ"));
    ASSERT_TRUE(it != end(attributes));
    ASSERT_TRUE(it->second == "zusätzlicher Zug");
  }
}

TEST(loader_hrd_attributes, ignore_output_rules) {
  for (auto const& c : configs) {
    loaded_file f = {c.files(ATTRIBUTES), "# ,  ,  ,"};
    ASSERT_TRUE(parse_attributes(f, c).empty());
  }
}

}  // namespace motis::loader::hrd
