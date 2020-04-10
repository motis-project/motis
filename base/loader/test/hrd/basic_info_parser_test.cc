#include "gtest/gtest.h"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/hrd/parser/basic_info_parser.h"

namespace motis::loader::hrd {

constexpr auto const ECDATEN_FILE_CONTENT =
    "14.12.2014\n"
    "12.12.2015\n"
    "JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END\n";

TEST(loader_hrd_basic_info, simple_interval) {
  auto const c = hrd_5_00_8;
  auto interval = parse_interval({c.files(BASIC_DATA), ECDATEN_FILE_CONTENT});
  EXPECT_EQ(1418515200, interval.from());
  EXPECT_EQ(1449878400, interval.to());
}

TEST(loader_hrd_basic_info, schedule_name) {
  auto const c = hrd_5_00_8;
  auto name = parse_schedule_name({c.files(BASIC_DATA), ECDATEN_FILE_CONTENT});
  EXPECT_EQ("JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END",
            name);
}

}  // namespace motis::loader::hrd
