#include "gtest/gtest.h"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include "nigiri/footpath.h"
#include "nigiri/td_footpath.h"

#include "motis/flex/mode_id.h"
#include "motis/td_offsets.h"

namespace n = nigiri;

namespace {

n::unixtime_t t(int const minutes) {
  return n::unixtime_t{n::i32_minutes{minutes}};
}

n::transport_mode_id_t flex_id(n::flex_transport_idx_t::value_t const transport,
                               n::stop_idx_t const stop,
                               osr::direction const dir) {
  return motis::flex::mode_id{n::flex_transport_idx_t{transport}, stop, dir}
      .to_id();
}

struct offer {
  int from_, to_;
  n::duration_t duration_;
  n::transport_mode_id_t mode_;
};

// Builds the raw input as the producers do: every offer contributes a start
// entry and a kMaxDuration closer, both tagged with the offer's mode.
std::vector<n::routing::td_offset> raw(std::initializer_list<offer> offers) {
  auto v = std::vector<n::routing::td_offset>{};
  for (auto const& o : offers) {
    v.push_back({.valid_from_ = t(o.from_),
                 .duration_ = o.duration_,
                 .transport_mode_id_ = o.mode_});
    v.push_back({.valid_from_ = t(o.to_),
                 .duration_ = n::footpath::kMaxDuration,
                 .transport_mode_id_ = o.mode_});
  }
  return v;
}

n::routing::td_offset active(int const from,
                             int const duration,
                             n::transport_mode_id_t const mode) {
  return {.valid_from_ = t(from),
          .duration_ = n::duration_t{duration},
          .transport_mode_id_ = mode};
}

n::routing::td_offset inactive(int const from) {
  return {.valid_from_ = t(from),
          .duration_ = n::footpath::kMaxDuration,
          .transport_mode_id_ = 0U};
}

// Arrival time the routing core (nigiri's get_td_duration) yields for a
// departure at minute `dep`; std::nullopt if no offer is reachable.
std::optional<n::unixtime_t> arrival(
    std::vector<n::routing::td_offset> const& offsets, int const dep) {
  auto const r = n::get_td_duration<n::direction::kForward>(
      std::span<n::routing::td_offset const>{offsets}, t(dep));
  return r.has_value() ? std::optional{t(dep) + r->first} : std::nullopt;
}

}  // namespace

TEST(motis, td_offsets_keep_shortest_same_window) {
  auto const slow = flex_id(1U, 0U, osr::direction::kBackward);
  auto const fast = flex_id(2U, 0U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 200, n::duration_t{30}, slow},
      {100, 200, n::duration_t{10}, fast},
  });
  motis::normalize_td_offsets(offsets);

  EXPECT_EQ((std::vector{inactive(0), active(100, 10, fast), inactive(200)}),
            offsets);
}

TEST(motis, td_offsets_deterministic_on_equal_duration) {
  auto const first = flex_id(1U, 0U, osr::direction::kBackward);
  auto const second = flex_id(2U, 0U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 200, n::duration_t{10}, first},
      {100, 200, n::duration_t{10}, second},
  });
  motis::normalize_td_offsets(offsets);

  EXPECT_EQ((std::vector{inactive(0), active(100, 10, std::min(first, second)),
                         inactive(200)}),
            offsets);
}

TEST(motis, td_offsets_split_overlapping_windows) {
  auto const slow = flex_id(1U, 0U, osr::direction::kBackward);
  auto const fast = flex_id(2U, 0U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 220, n::duration_t{30}, slow},
      {150, 180, n::duration_t{10}, fast},
  });
  motis::normalize_td_offsets(offsets);

  // The slow offer is ended early at 150 + 10 - 30 = 130: departing later than
  // that, it is faster to wait for the fast offer (FIFO repair). Without it,
  // departing at 149 (arrival 179) would beat departing at 150 (arrival 160).
  EXPECT_EQ((std::vector{inactive(0), active(100, 30, slow), inactive(130),
                         active(150, 10, fast), active(180, 30, slow),
                         inactive(220)}),
            offsets);

  // get_td_duration must yield FIFO arrivals. [130, 150) is the cut region
  // where the slow offer would otherwise let an earlier departure overtake:
  EXPECT_EQ(t(130), arrival(offsets, 100));
  EXPECT_EQ(t(159), arrival(offsets, 129));
  for (auto dep = 130; dep < 150; ++dep) {
    EXPECT_EQ(t(160), arrival(offsets, dep)) << "at minute " << dep;
  }
  EXPECT_EQ(t(160), arrival(offsets, 150));
  EXPECT_EQ(t(189), arrival(offsets, 179));
  EXPECT_EQ(t(210), arrival(offsets, 180));
  EXPECT_EQ(t(249), arrival(offsets, 219));
}

TEST(motis, td_offsets_fifo_cascade) {
  auto const a = flex_id(1U, 0U, osr::direction::kBackward);
  auto const b = flex_id(2U, 0U, osr::direction::kBackward);
  auto const c = flex_id(3U, 0U, osr::direction::kBackward);

  // Three overlapping offers, each faster than the previous: the FIFO repair
  // cuts `a` relative to `b` (at 200 + 60 - 90 = 170) and `b` relative to `c`
  // (at 300 + 10 - 60 = 250).
  auto offsets = raw({
      {100, 400, n::duration_t{90}, a},
      {200, 400, n::duration_t{60}, b},
      {300, 400, n::duration_t{10}, c},
  });
  motis::normalize_td_offsets(offsets);

  EXPECT_EQ((std::vector{inactive(0), active(100, 90, a), inactive(170),
                         active(200, 60, b), inactive(250), active(300, 10, c),
                         inactive(400)}),
            offsets);

  // get_td_duration must yield FIFO arrivals across both cut regions:
  EXPECT_EQ(t(259), arrival(offsets, 169));
  for (auto dep = 170; dep < 200; ++dep) {
    EXPECT_EQ(t(260), arrival(offsets, dep)) << "at minute " << dep;
  }
  EXPECT_EQ(t(309), arrival(offsets, 249));
  for (auto dep = 250; dep < 300; ++dep) {
    EXPECT_EQ(t(310), arrival(offsets, dep)) << "at minute " << dep;
  }
  EXPECT_EQ(t(409), arrival(offsets, 399));
}

TEST(motis, td_offsets_keep_inactive_gaps) {
  auto const first = flex_id(1U, 0U, osr::direction::kBackward);
  auto const second = flex_id(2U, 0U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 120, n::duration_t{10}, first},
      {150, 170, n::duration_t{8}, second},
  });
  motis::normalize_td_offsets(offsets);

  // No FIFO violation: departing at 119 arrives at 129, departing at 150
  // arrives at 158 -> the slow offer is kept as-is.
  EXPECT_EQ((std::vector{inactive(0), active(100, 10, first), inactive(120),
                         active(150, 8, second), inactive(170)}),
            offsets);
}

TEST(motis, td_offsets_drop_fully_dominated_window) {
  auto const slow = flex_id(1U, 0U, osr::direction::kBackward);
  auto const fast = flex_id(2U, 0U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 200, n::duration_t{100}, slow},
      {150, 250, n::duration_t{10}, fast},
  });
  motis::normalize_td_offsets(offsets);

  // Waiting from 100 for the fast offer (arrival 160) always beats the slow
  // offer (arrival >= 200) -> the slow window is fully replaced by an
  // inactive gap.
  EXPECT_EQ((std::vector{inactive(0), active(150, 10, fast), inactive(250)}),
            offsets);

  // The whole slow window is dominated: every departure in [100, 150) arrives
  // no earlier than waiting for the fast offer (arrival 160) would.
  for (auto dep = 100; dep < 150; ++dep) {
    EXPECT_EQ(t(160), arrival(offsets, dep)) << "at minute " << dep;
  }
  EXPECT_EQ(t(160), arrival(offsets, 150));
  EXPECT_EQ(t(259), arrival(offsets, 249));
}

TEST(motis, td_offsets_merge_inactive_after_cut) {
  auto const slow = flex_id(1U, 0U, osr::direction::kBackward);
  auto const fast = flex_id(2U, 0U, osr::direction::kBackward);

  // The slow offer is cut at 250 + 10 - 100 = 160; the envelope already has an
  // inactive gap at 200 (slow closes, fast not yet open). The cut's inactive
  // entry and that gap are adjacent kMaxDuration entries -> they must collapse
  // into a single one.
  auto offsets = raw({
      {100, 200, n::duration_t{100}, slow},
      {250, 400, n::duration_t{10}, fast},
  });
  motis::normalize_td_offsets(offsets);

  EXPECT_EQ((std::vector{inactive(0), active(100, 100, slow), inactive(160),
                         active(250, 10, fast), inactive(400)}),
            offsets);

  // Verify via nigiri's get_td_duration that the result is FIFO. Before the
  // cut the slow offer is used directly:
  EXPECT_EQ(t(200), arrival(offsets, 100));
  EXPECT_EQ(t(259), arrival(offsets, 159));
  // During the cut/overlap region departing later must never arrive earlier:
  // the routing core waits for the fast offer (arrival 260) instead of taking
  // the slow one, which would arrive at dep + 100 > 260 (the non-FIFO case).
  for (auto dep = 160; dep < 250; ++dep) {
    EXPECT_EQ(t(260), arrival(offsets, dep)) << "at minute " << dep;
  }
  // Inside the fast window the arrival grows monotonically again:
  EXPECT_EQ(t(260), arrival(offsets, 250));
  EXPECT_EQ(t(310), arrival(offsets, 300));
}

TEST(motis, td_offsets_preserve_mode_id) {
  auto const backward = flex_id(42U, 3U, osr::direction::kBackward);

  auto offsets = raw({
      {100, 200, n::duration_t{10}, backward},
  });
  motis::normalize_td_offsets(offsets);

  ASSERT_EQ(3U, offsets.size());
  auto const restored = motis::flex::mode_id{offsets[1].transport_mode_id_};
  EXPECT_EQ(osr::direction::kBackward, restored.get_dir());
  EXPECT_EQ(n::flex_transport_idx_t{42U}, restored.get_flex_transport());
  EXPECT_EQ(static_cast<n::stop_idx_t>(3U), restored.get_stop());
}
