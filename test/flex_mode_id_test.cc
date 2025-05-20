#include "gtest/gtest.h"

#include "motis/flex/mode_id.h"

using namespace motis::flex;

TEST(motis, flex_mode_id_zero) {
  auto const t = nigiri::flex_transport_idx_t{0U};
  auto const stop = 0U;
  auto const dir = osr::direction::kForward;
  auto const id = mode_id{t, stop, dir}.to_id();
  EXPECT_TRUE(mode_id::is_flex(id));

  auto const id1 = mode_id{id};
  EXPECT_EQ(stop, id1.get_stop());
  EXPECT_EQ(dir, id1.get_dir());
  EXPECT_EQ(t, id1.get_flex_transport());
}

TEST(motis, flex_mode_id) {
  auto const t = nigiri::flex_transport_idx_t{44444U};
  auto const stop = 15;
  auto const dir = osr::direction::kBackward;
  auto const id = mode_id{t, stop, dir}.to_id();
  EXPECT_TRUE(mode_id::is_flex(id));

  auto const id1 = mode_id{id};
  EXPECT_EQ(stop, id1.get_stop());
  EXPECT_EQ(dir, id1.get_dir());
  EXPECT_EQ(t, id1.get_flex_transport());
}