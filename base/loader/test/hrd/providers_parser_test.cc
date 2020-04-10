#include "gtest/gtest.h"

#include "motis/loader/hrd/parser/providers_parser.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

TEST(loader_hrd_providers, simple) {
  char const* file_content =
      "00001 K 'DPN' L 'ABR' V 'ABELLIO Rail Mitteldeutschland GmbH'\n"
      "00001 : AM____\n"
      "00002 K 'DPN' L 'ABR' V 'ABELLIO Rail NRW GmbH'\n"
      "00002 : AR____\n"
      "00003 K 'DPN' L 'ag ' V 'agilis'\n"
      "00003 : A9____ XY____\n";
  for (auto const& c : configs) {
    auto providers = parse_providers({"betrieb.101", file_content}, c);

    EXPECT_EQ(4U, providers.size());

    auto const& first = providers[raw_to_int<uint64_t>("AM____")];
    EXPECT_EQ("DPN", first.short_name_);
    EXPECT_EQ("ABR", first.long_name_);
    EXPECT_EQ("ABELLIO Rail Mitteldeutschland GmbH", first.full_name_);

    auto const& second = providers[raw_to_int<uint64_t>("AR____")];
    EXPECT_EQ("DPN", second.short_name_);
    EXPECT_EQ("ABR", second.long_name_);
    EXPECT_EQ("ABELLIO Rail NRW GmbH", second.full_name_);

    auto const& third = providers[raw_to_int<uint64_t>("A9____")];
    EXPECT_EQ("DPN", third.short_name_);
    EXPECT_EQ("ag ", third.long_name_);
    EXPECT_EQ("agilis", third.full_name_);

    auto const& fourth = providers[raw_to_int<uint64_t>("XY____")];
    EXPECT_EQ("DPN", fourth.short_name_);
    EXPECT_EQ("ag ", fourth.long_name_);
    EXPECT_EQ("agilis", fourth.full_name_);
  }
}

}  // namespace motis::loader::hrd
