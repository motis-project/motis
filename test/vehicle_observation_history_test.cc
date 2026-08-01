#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"

#include "motis/rt/vehicle_observation_history.h"

namespace motis {
namespace {

using namespace std::chrono_literals;

constexpr auto const kPolicy = observation_history_policy{
    .max_age_ = 300s, .max_observations_per_vehicle_ = 3U};

vehicle_observation observation(
    std::int64_t const ingested,
    std::optional<std::int64_t> const reported = std::nullopt,
    std::string entity = "entity",
    std::optional<std::string> vehicle = std::optional<std::string>{"vehicle"},
    std::string trip = "trip") {
  return vehicle_observation{
      .feed_id_ = "feed",
      .entity_id_ = std::move(entity),
      .vehicle_id_ = std::move(vehicle),
      .trip_ = {.trip_id_ = std::move(trip), .start_date_ = "20260731"},
      .latitude_ = 50.0,
      .longitude_ = 19.0 + static_cast<double>(ingested) / 1000.0,
      .reported_time_ = reported,
      .ingested_time_ = ingested};
}

vehicle_key descriptor_key(std::string id = "vehicle") {
  return vehicle_key{"feed", std::move(id),
                     vehicle_key_source::kVehicleDescriptor};
}

vehicle_key entity_key(std::string id = "entity") {
  return vehicle_key{"feed", std::move(id), vehicle_key_source::kEntityId};
}

TEST(vehicle_observation_history, orders_by_reported_then_ingest_time) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(120, 110), kPolicy));
  EXPECT_TRUE(history.ingest(observation(130, 100), kPolicy));
  EXPECT_TRUE(history.ingest(observation(125, 110), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 3U);
  EXPECT_EQ(values[0].reported_time_, 100);
  EXPECT_EQ(values[1].ingested_time_, 120);
  EXPECT_EQ(values[2].ingested_time_, 125);
  ASSERT_NE(history.effective_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.effective_observation(descriptor_key())->ingested_time_,
            125);
}

TEST(vehicle_observation_history,
     falls_back_to_ingest_time_when_reported_time_is_missing) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(100), kPolicy));
  EXPECT_TRUE(history.ingest(observation(120, 110), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(observation_time(values[0]), 100);
  EXPECT_EQ(observation_time(values[1]), 110);
}

TEST(vehicle_observation_history,
     deduplicates_repeated_reports_without_refreshing_their_age) {
  auto history = vehicle_observation_history{};
  auto const first = observation(100, 90);
  auto repeated = first;
  repeated.ingested_time_ = 200;

  EXPECT_TRUE(history.ingest(first, kPolicy));
  EXPECT_TRUE(history.ingest(repeated, kPolicy));
  ASSERT_EQ(history.observation_count(), 1U);
  EXPECT_EQ(history.observations(descriptor_key())[0].ingested_time_, 200);

  history.prune(391, kPolicy);
  EXPECT_EQ(history.observation_count(), 0U);
}

TEST(vehicle_observation_history, enforces_count_and_age_bounds) {
  auto history = vehicle_observation_history{};
  for (auto const time : {100, 110, 120, 130}) {
    EXPECT_TRUE(history.ingest(observation(time, time), kPolicy));
  }

  auto values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 3U);
  EXPECT_EQ(values.front().reported_time_, 110);

  history.prune(421, kPolicy);
  values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 1U);
  EXPECT_EQ(values.front().reported_time_, 130);
}

TEST(vehicle_observation_history, resets_when_trip_instance_changes) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(90, 90), kPolicy));
  EXPECT_TRUE(history.ingest(
      observation(100, 100, "entity", "vehicle", "next-trip"), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 1U);
  EXPECT_EQ(values.front().trip_.trip_id_, "next-trip");
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->trip_.trip_id_,
            "next-trip");
}

TEST(vehicle_observation_history,
     ignores_late_observation_from_prior_trip_instance) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(
      observation(200, 200, "entity", "vehicle", "next-trip"), kPolicy));
  EXPECT_TRUE(history.ingest(observation(210, 150), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 1U);
  EXPECT_EQ(values.front().trip_.trip_id_, "next-trip");
  ASSERT_NE(history.effective_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.effective_observation(descriptor_key())->reported_time_,
            200);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
}

TEST(vehicle_observation_history, resets_when_identity_key_source_changes) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(100, 100), kPolicy));
  EXPECT_TRUE(
      history.ingest(observation(110, 110, "entity", std::nullopt), kPolicy));

  EXPECT_TRUE(history.observations(descriptor_key()).empty());
  ASSERT_EQ(history.observations(entity_key()).size(), 1U);
  EXPECT_EQ(history.active_histories(), 1U);
}

TEST(vehicle_observation_history,
     ignores_late_observation_missing_the_vehicle_descriptor_id) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(200, 200), kPolicy));
  EXPECT_TRUE(
      history.ingest(observation(210, 150, "entity", std::nullopt), kPolicy));

  ASSERT_EQ(history.observations(descriptor_key()).size(), 1U);
  EXPECT_TRUE(history.observations(entity_key()).empty());
  ASSERT_NE(history.effective_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.effective_observation(descriptor_key())->reported_time_,
            200);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
}

TEST(vehicle_observation_history,
     ignores_late_vehicle_descriptor_after_entity_id_fallback) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(
      history.ingest(observation(200, 200, "entity", std::nullopt), kPolicy));
  EXPECT_TRUE(history.ingest(observation(210, 150), kPolicy));

  ASSERT_EQ(history.observations(entity_key()).size(), 1U);
  EXPECT_TRUE(history.observations(descriptor_key()).empty());
  ASSERT_NE(history.effective_observation(entity_key()), nullptr);
  EXPECT_EQ(history.effective_observation(entity_key())->reported_time_, 200);
  ASSERT_NE(history.current_observation(entity_key()), nullptr);
  EXPECT_EQ(history.current_observation(entity_key())->reported_time_, 200);
}

TEST(vehicle_observation_history,
     ignores_late_observation_for_a_different_vehicle_descriptor) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(200, 200), kPolicy));
  EXPECT_TRUE(history.ingest(observation(210, 150, "entity", "other-vehicle"),
                             kPolicy));

  ASSERT_EQ(history.observations(descriptor_key()).size(), 1U);
  EXPECT_TRUE(history.observations(descriptor_key("other-vehicle")).empty());
  ASSERT_NE(history.effective_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.effective_observation(descriptor_key())->reported_time_,
            200);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
}

TEST(vehicle_observation_history,
     keeps_history_when_only_the_feed_entity_id_changes) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(100, 100, "first-entity"), kPolicy));
  EXPECT_TRUE(history.ingest(observation(110, 110, "second-entity"), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values.front().entity_id_, "first-entity");
  EXPECT_EQ(values.back().entity_id_, "second-entity");
}

TEST(vehicle_observation_history,
     retains_out_of_order_data_without_regressing_effective_observation) {
  auto history = vehicle_observation_history{};
  EXPECT_TRUE(history.ingest(observation(200, 200), kPolicy));
  EXPECT_TRUE(history.ingest(observation(210, 150), kPolicy));

  auto const values = history.observations(descriptor_key());
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values.front().reported_time_, 150);
  ASSERT_NE(history.effective_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.effective_observation(descriptor_key())->reported_time_,
            200);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
}

TEST(vehicle_observation_history,
     full_replacement_invalidates_current_but_preserves_short_history) {
  auto history = vehicle_observation_history{};
  auto const initial = std::array{observation(100, 100)};
  history.replace_feed("feed", initial, 100, kPolicy);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);

  history.replace_feed("feed", {}, 110, kPolicy);

  EXPECT_EQ(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.observations(descriptor_key()).size(), 1U);
  history.prune(401, kPolicy);
  EXPECT_TRUE(history.observations(descriptor_key()).empty());
}

TEST(vehicle_observation_history,
     full_replacement_keeps_newer_current_across_a_late_prior_trip) {
  auto history = vehicle_observation_history{};
  auto const current =
      std::array{observation(200, 200, "entity", "vehicle", "next-trip")};
  history.replace_feed("feed", current, 200, kPolicy);

  auto const late = std::array{observation(210, 150)};
  history.replace_feed("feed", late, 210, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
  EXPECT_EQ(history.current_observation(descriptor_key())->trip_.trip_id_,
            "next-trip");
  ASSERT_EQ(history.observations(descriptor_key()).size(), 1U);
}

TEST(vehicle_observation_history,
     full_replacement_keeps_newer_current_across_a_late_identity_change) {
  auto history = vehicle_observation_history{};
  auto const current = std::array{observation(200, 200)};
  history.replace_feed("feed", current, 200, kPolicy);

  auto const late = std::array{observation(210, 150, "entity", std::nullopt)};
  history.replace_feed("feed", late, 210, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
  EXPECT_EQ(history.current_observation(entity_key()), nullptr);
}

TEST(vehicle_observation_history,
     full_replacement_removes_only_current_vehicles_that_are_absent) {
  auto history = vehicle_observation_history{};
  auto const current =
      std::array{observation(200, 200),
                 observation(200, 200, "other-entity", "other-vehicle")};
  history.replace_feed("feed", current, 200, kPolicy);

  auto const late = std::array{observation(210, 150)};
  history.replace_feed("feed", late, 210, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key())->reported_time_, 200);
  EXPECT_EQ(history.current_observation(descriptor_key("other-vehicle")),
            nullptr);
  EXPECT_EQ(history.observations(descriptor_key("other-vehicle")).size(), 1U);
}

TEST(vehicle_observation_history,
     full_replacement_preserves_vehicle_when_its_old_entity_is_reused) {
  auto history = vehicle_observation_history{};
  auto const initial = std::array{observation(100, 100, "e1", "V")};
  history.replace_feed("feed", initial, 100, kPolicy);

  auto const replacement = std::array{observation(110, 110, "e2", "V"),
                                      observation(110, 110, "e1", "W")};
  history.replace_feed("feed", replacement, 110, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key("V")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("V"))->entity_id_, "e2");
  ASSERT_EQ(history.observations(descriptor_key("V")).size(), 2U);
  ASSERT_NE(history.current_observation(descriptor_key("W")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("W"))->entity_id_, "e1");
}

TEST(vehicle_observation_history,
     full_replacement_is_order_independent_when_old_entity_is_reused) {
  auto history = vehicle_observation_history{};
  auto const initial = std::array{observation(100, 100, "e1", "V")};
  history.replace_feed("feed", initial, 100, kPolicy);

  auto const replacement = std::array{observation(110, 110, "e1", "W"),
                                      observation(110, 110, "e2", "V")};
  history.replace_feed("feed", replacement, 110, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key("V")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("V"))->entity_id_, "e2");
  ASSERT_EQ(history.observations(descriptor_key("V")).size(), 2U);
  ASSERT_NE(history.current_observation(descriptor_key("W")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("W"))->entity_id_, "e1");
}

TEST(vehicle_observation_history,
     differential_reuse_of_old_entity_preserves_rotated_vehicle) {
  auto history = vehicle_observation_history{};
  auto const first = std::array{observation(100, 100, "e1", "V")};
  history.update_feed("feed", first, {}, 100, kPolicy);
  auto const rotated = std::array{observation(110, 110, "e2", "V")};
  history.update_feed("feed", rotated, {}, 110, kPolicy);

  auto const reused = std::array{observation(120, 120, "e1", "W")};
  history.update_feed("feed", reused, {}, 120, kPolicy);

  ASSERT_NE(history.current_observation(descriptor_key("V")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("V"))->entity_id_, "e2");
  ASSERT_EQ(history.observations(descriptor_key("V")).size(), 2U);
  ASSERT_NE(history.current_observation(descriptor_key("W")), nullptr);
  EXPECT_EQ(history.current_observation(descriptor_key("W"))->entity_id_, "e1");
}

TEST(vehicle_observation_history,
     differential_deletion_invalidates_current_but_preserves_short_history) {
  auto history = vehicle_observation_history{};
  auto const initial = std::array{observation(100, 100)};
  history.update_feed("feed", initial, {}, 100, kPolicy);
  ASSERT_NE(history.current_observation(descriptor_key()), nullptr);

  auto const deleted = std::array<std::string, 1>{"entity"};
  history.update_feed("feed", {}, deleted, 110, kPolicy);

  EXPECT_EQ(history.current_observation(descriptor_key()), nullptr);
  EXPECT_EQ(history.observations(descriptor_key()).size(), 1U);
  history.prune(401, kPolicy);
  EXPECT_TRUE(history.observations(descriptor_key()).empty());
}

TEST(vehicle_observation_history, rejects_observations_without_any_identity) {
  auto history = vehicle_observation_history{};
  EXPECT_FALSE(history.ingest(
      observation(100, 100, "", std::optional<std::string>{""}), kPolicy));
  EXPECT_EQ(history.active_histories(), 0U);
}

}  // namespace
}  // namespace motis
