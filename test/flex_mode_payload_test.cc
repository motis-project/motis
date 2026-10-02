#include "gtest/gtest.h"

#include "motis/flex/mode_payload.h"

namespace n = nigiri;

using namespace motis::flex;

TEST(motis, flex_mode_payload_zero) {
  auto const t = n::flex_transport_idx_t{0U};
  auto const from = n::stop_idx_t{0U};
  auto const to = n::stop_idx_t{0U};
  auto const payload = mode_payload{t, from, to}.to_payload();

  auto const p = mode_payload{payload};
  EXPECT_EQ(from, p.get_from_stop());
  EXPECT_EQ(to, p.get_to_stop());
  EXPECT_EQ(t, p.get_flex_transport());
}

TEST(motis, flex_mode_payload) {
  auto const t = n::flex_transport_idx_t{(1U << 18U) - 1U};
  auto const from = n::stop_idx_t{15U};
  auto const to = n::stop_idx_t{127U};
  ASSERT_TRUE(mode_payload::fits(t, from, to));
  auto const payload = mode_payload{t, from, to}.to_payload();

  auto const p = mode_payload{payload};
  EXPECT_EQ(from, p.get_from_stop());
  EXPECT_EQ(to, p.get_to_stop());
  EXPECT_EQ(t, p.get_flex_transport());
}

TEST(motis, flex_mode_payload_fits) {
  EXPECT_FALSE(mode_payload::fits(n::flex_transport_idx_t{1U << 18U}, 0U, 1U));
  EXPECT_FALSE(mode_payload::fits(n::flex_transport_idx_t{0U}, 0U, 128U));
  EXPECT_FALSE(mode_payload::fits(n::flex_transport_idx_t{0U}, 128U, 1U));
}
