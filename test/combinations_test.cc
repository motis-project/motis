#include "gtest/gtest.h"

#include "date/date.h"

#include "nigiri/types.h"

#include "motis/elevators/get_state_changes.h"

using namespace date;
using namespace std::chrono_literals;
using namespace motis;
namespace n = nigiri;

std::ostream& operator<<(std::ostream& out, std::vector<bool> const& v) {
  auto first = true;
  for (auto const b : v) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << std::boolalpha << b;
  }
  return out;
}

TEST(motis, int_state_changes) {
  auto const changes = std::vector<std::vector<state_change<int>>>{
      {{{0, true}, {10, false}, {20, true}}},
      {{{0, false},
        {5, true},
        {10, false},
        {15, true},
        {20, false},
        {25, true},
        {30, false}}}};
  auto g = get_state_changes(changes);
  auto const expected = std::array<std::pair<int, std::vector<bool>>, 7>{
      std::pair<int, std::vector<bool>>{0, {true, false}},
      std::pair<int, std::vector<bool>>{5, {true, true}},
      std::pair<int, std::vector<bool>>{10, {false, false}},
      std::pair<int, std::vector<bool>>{15, {false, true}},
      std::pair<int, std::vector<bool>>{20, {true, false}},
      std::pair<int, std::vector<bool>>{25, {true, true}},
      std::pair<int, std::vector<bool>>{30, {true, false}}};
  auto i = 0U;
  while (g) {
    EXPECT_EQ(expected[i++], g());
  }
  EXPECT_EQ(i, expected.size());
}