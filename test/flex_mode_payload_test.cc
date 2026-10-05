#include "gtest/gtest.h"

#include "motis/flex/mode_payload.h"

namespace n = nigiri;

using namespace motis::flex;

TEST(motis, flex_mode_payload_zero) {
  auto const t = n::flex_transport_idx_t{0U};
  auto const stop = 0U;
  auto const dir = osr::direction::kForward;
  auto const payload = mode_payload{t, stop, dir}.to_payload();

  auto const p = mode_payload{payload};
  EXPECT_EQ(stop, p.get_stop());
  EXPECT_EQ(dir, p.get_dir());
  EXPECT_EQ(t, p.get_flex_transport());
}

TEST(motis, flex_mode_payload) {
  auto const t = n::flex_transport_idx_t{44444U};
  auto const stop = 15;
  auto const dir = osr::direction::kBackward;
  auto const payload = mode_payload{t, stop, dir}.to_payload();

  auto const p = mode_payload{payload};
  EXPECT_EQ(stop, p.get_stop());
  EXPECT_EQ(dir, p.get_dir());
  EXPECT_EQ(t, p.get_flex_transport());
}
