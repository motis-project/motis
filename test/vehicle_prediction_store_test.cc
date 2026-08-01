#include "gtest/gtest.h"

#include "motis/rt/vehicle_prediction_store.h"

namespace motis {
namespace {

auto entry(unsigned const transport, unsigned const stop, std::int64_t seen) {
  return vehicle_prediction_diagnostic_entry{
      .transport_ = nigiri::transport{nigiri::transport_idx_t{transport},
                                      nigiri::day_idx_t{0U}},
      .static_stop_sequence_ = stop,
      .trip_id_ = "feed_trip",
      .observed_at_seconds_ = seen,
      .effective_ = {.predicted_timestamp_seconds_ =
                         static_cast<std::int64_t>(stop) * 100,
                     .delay_seconds_ = 10}};
}

TEST(vehicle_prediction_store, allocates_nothing_when_disabled) {
  EXPECT_EQ(nullptr, vehicle_prediction_diagnostics_store::build(
                         false, {entry(1U, 2U, 100)}, 100));
}

TEST(vehicle_prediction_store, bounds_expires_deduplicates_and_finds) {
  auto store = vehicle_prediction_diagnostics_store::build(
      true,
      {entry(2U, 4U, 95), entry(1U, 3U, 99), entry(1U, 3U, 98),
       entry(3U, 5U, 1)},
      100, {.max_age_seconds_ = 10, .max_entries_ = 2U});
  ASSERT_NE(nullptr, store);
  ASSERT_EQ(2U, store->size());
  ASSERT_NE(nullptr, store->find(
                         nigiri::transport{nigiri::transport_idx_t{1U},
                                           nigiri::day_idx_t{0U}},
                         3U));
  EXPECT_EQ(99, store->find(nigiri::transport{nigiri::transport_idx_t{1U},
                                              nigiri::day_idx_t{0U}},
                            3U)
                    ->observed_at_seconds_);
  EXPECT_EQ(nullptr, store->find(
                         nigiri::transport{nigiri::transport_idx_t{3U},
                                           nigiri::day_idx_t{0U}},
                         5U));
  EXPECT_NE(nullptr,
            store->find_event(
                nigiri::transport{nigiri::transport_idx_t{1U},
                                  nigiri::day_idx_t{0U}},
                "feed_trip", 290));
}

}  // namespace
}  // namespace motis
