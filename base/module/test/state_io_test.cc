#include "gtest/gtest.h"

#include "motis/module/ini_io.h"

using namespace motis::module;

struct test_state {
  named<std::string, MOTIS_NAME("path")> path_;
  named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  named<size_t, MOTIS_NAME("size")> size_;
};

constexpr auto const file_content = R"(path=test
hash=12345
size=54321
)";

TEST(motis_module_state, write) {
  test_state s{std::string{"test"}, 12345ULL, 54321U};
  std::stringstream ss;
  write_ini(ss, s);
  EXPECT_EQ(ss.str(), file_content);
}

TEST(motis_module_state, read) {
  auto const r = read_ini<test_state>(std::string{file_content});
  EXPECT_EQ(r.path_.val(), "test");
  EXPECT_EQ(r.hash_.val(), 12345U);
  EXPECT_EQ(r.size_.val(), 54321U);
}