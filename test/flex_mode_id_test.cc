#include "gtest/gtest.h"

#include "motis/flex/flex.h"
#include "motis/flex/mode_id.h"

namespace n = nigiri;

using namespace motis::flex;

namespace {

n::unixtime_t t(int const minutes) {
  return n::unixtime_t{n::i32_minutes{minutes}};
}

n::transport_mode_id_t flex_id(n::flex_transport_idx_t::value_t const transport,
                               n::stop_idx_t const stop,
                               osr::direction const dir) {
  return mode_id{n::flex_transport_idx_t{transport}, stop, dir}.to_id();
}

}  // namespace

TEST(motis, flex_mode_id_zero) {
  auto const t = n::flex_transport_idx_t{0U};
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
  auto const t = n::flex_transport_idx_t{44444U};
  auto const stop = 15;
  auto const dir = osr::direction::kBackward;
  auto const id = mode_id{t, stop, dir}.to_id();
  EXPECT_TRUE(mode_id::is_flex(id));

  auto const id1 = mode_id{id};
  EXPECT_EQ(stop, id1.get_stop());
  EXPECT_EQ(dir, id1.get_dir());
  EXPECT_EQ(t, id1.get_flex_transport());
}

TEST(motis, flex_td_offsets_keep_shortest_same_window) {
  auto const slow = flex_id(1U, 0U, osr::direction::kBackward);
  auto const fast = flex_id(2U, 0U, osr::direction::kBackward);

  auto const offsets = normalize_flex_td_offsets({
      {t(100), t(200), n::duration_t{30}, slow},
      {t(100), t(200), n::duration_t{10}, fast},
  });

  ASSERT_EQ(3U, offsets.size());
  EXPECT_EQ(t(0), offsets[0].valid_from_);
  EXPECT_EQ(n::footpath::kMaxDuration, offsets[0].duration_);
  EXPECT_EQ(t(100), offsets[1].valid_from_);
  EXPECT_EQ(n::duration_t{10}, offsets[1].duration_);
  EXPECT_EQ(fast, offsets[1].transport_mode_id_);
  EXPECT_EQ(t(200), offsets[2].valid_from_);
  EXPECT_EQ(n::footpath::kMaxDuration, offsets[2].duration_);
}

TEST(motis, flex_td_offsets_preserve_first_on_equal_duration) {
  auto const first = flex_id(1U, 0U, osr::direction::kBackward);
  auto const second = flex_id(2U, 0U, osr::direction::kBackward);

  auto const offsets = normalize_flex_td_offsets({
      {t(100), t(200), n::duration_t{10}, first},
      {t(100), t(200), n::duration_t{10}, second},
  });

  ASSERT_EQ(3U, offsets.size());
  EXPECT_EQ(first, offsets[1].transport_mode_id_);
}

TEST(motis, flex_td_offsets_split_overlapping_windows) {
  auto const long_id = flex_id(1U, 0U, osr::direction::kBackward);
  auto const short_id = flex_id(2U, 0U, osr::direction::kBackward);

  auto const offsets = normalize_flex_td_offsets({
      {t(100), t(220), n::duration_t{30}, long_id},
      {t(150), t(180), n::duration_t{10}, short_id},
  });

  ASSERT_EQ(5U, offsets.size());
  EXPECT_EQ(t(100), offsets[1].valid_from_);
  EXPECT_EQ(long_id, offsets[1].transport_mode_id_);
  EXPECT_EQ(t(150), offsets[2].valid_from_);
  EXPECT_EQ(short_id, offsets[2].transport_mode_id_);
  EXPECT_EQ(t(180), offsets[3].valid_from_);
  EXPECT_EQ(long_id, offsets[3].transport_mode_id_);
  EXPECT_EQ(t(220), offsets[4].valid_from_);
  EXPECT_EQ(n::footpath::kMaxDuration, offsets[4].duration_);
}

TEST(motis, flex_td_offsets_keep_inactive_gaps) {
  auto const first = flex_id(1U, 0U, osr::direction::kBackward);
  auto const second = flex_id(2U, 0U, osr::direction::kBackward);

  auto const offsets = normalize_flex_td_offsets({
      {t(100), t(120), n::duration_t{10}, first},
      {t(150), t(170), n::duration_t{8}, second},
  });

  ASSERT_EQ(5U, offsets.size());
  EXPECT_EQ(t(120), offsets[2].valid_from_);
  EXPECT_EQ(n::footpath::kMaxDuration, offsets[2].duration_);
  EXPECT_EQ(t(150), offsets[3].valid_from_);
  EXPECT_EQ(second, offsets[3].transport_mode_id_);
}

TEST(motis, flex_td_offsets_preserve_mode_id_direction) {
  auto const backward = flex_id(42U, 3U, osr::direction::kBackward);

  auto const offsets = normalize_flex_td_offsets({
      {t(100), t(200), n::duration_t{10}, backward},
  });

  ASSERT_EQ(3U, offsets.size());
  auto const restored = mode_id{offsets[1].transport_mode_id_};
  EXPECT_EQ(osr::direction::kBackward, restored.get_dir());
  EXPECT_EQ(n::flex_transport_idx_t{42U}, restored.get_flex_transport());
  EXPECT_EQ(static_cast<n::stop_idx_t>(3U), restored.get_stop());
}
