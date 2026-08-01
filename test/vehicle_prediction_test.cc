#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nigiri/rt/frun.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"
#include "motis/rt/trip_progress_projection.h"
#include "motis/rt/vehicle_position.h"
#include "motis/rt/vehicle_prediction.h"
#include "motis/rt/vehicle_prediction_diagnostics.h"
#include "motis/tag_lookup.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {
namespace {

constexpr auto const kGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.0000,8.0000
B,B,50.0000,8.0100
C,C,50.0000,8.0200

# routes.txt
route_id,agency_id,route_short_name,route_type
route,Test,1,3

# trips.txt
route_id,service_id,trip_id,shape_id
route,S1,straight,straight-shape

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
straight,01:00:00,01:00:00,A,10
straight,01:05:00,01:05:00,B,20
straight,01:10:00,01:10:00,C,30

# calendar_dates.txt
service_id,date,exception_type
S1,20260521,1

# shapes.txt
shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
straight-shape,50.0000,8.0000,1
straight-shape,50.0000,8.0050,2
straight-shape,50.0000,8.0100,3
straight-shape,50.0000,8.0150,4
straight-shape,50.0000,8.0200,5
)";

struct prediction_fixture {
  prediction_fixture()
      : path_{fs::temp_directory_path() / "motis-vehicle-prediction-test"},
        config_{.timetable_ = config::timetable{
                    .first_day_ = "2026-05-21",
                    .num_days_ = 2,
                    .with_shapes_ = true,
                    .datasets_ = {{"prediction", {.path_ = kGtfs}}}}} {
    auto ec = std::error_code{};
    fs::remove_all(path_, ec);
    import(config_, path_);
    data_.emplace(path_, config_);
  }

  ~prediction_fixture() {
    data_.reset();
    auto ec = std::error_code{};
    fs::remove_all(path_, ec);
  }

  n::rt::frun run() const {
    auto const [run, trip] = data_->tags_->get_trip(
        *data_->tt_, nullptr, "20260521_01:00_prediction_straight");
    EXPECT_TRUE(run.valid());
    EXPECT_NE(trip, n::trip_idx_t::invalid());
    return n::rt::frun{*data_->tt_, nullptr, run};
  }

  std::int64_t next_scheduled() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
               run()[1].scheduled_time(n::event_type::kArr).time_since_epoch())
        .count();
  }

  vehicle_observation observation(
      double const longitude,
      std::int64_t const timestamp,
      std::optional<double> const speed = std::nullopt,
      unsigned const next_stop = 20U,
      std::string status = "IN_TRANSIT_TO") const {
    return {.feed_id_ = "prediction:vp",
            .entity_id_ = "entity",
            .vehicle_id_ = "vehicle",
            .trip_ = {.trip_id_ = "straight",
                      .start_date_ = "20260521",
                      .start_time_ = "01:00:00"},
            .latitude_ = 50.0,
            .longitude_ = longitude,
            .speed_mps_ = speed,
            .current_stop_sequence_ = next_stop,
            .current_status_ = std::move(status),
            .reported_time_ = timestamp,
            .ingested_time_ = timestamp};
  }

  fs::path path_;
  config config_;
  std::optional<data> data_;
};

vehicle_prediction_batch evaluate_at_offset(prediction_fixture& fixture,
                                            std::int64_t const offset) {
  auto const now = fixture.next_scheduled() - 50 + offset;
  auto observations = std::vector<vehicle_observation>{
      fixture.observation(8.004, now - 10), fixture.observation(8.005, now)};
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};
  return engine.evaluate(fixture.run(), observations, now);
}

TEST(vehicle_prediction, produces_second_precision_whole_trip_candidates) {
  auto fixture = prediction_fixture{};

  auto const on_time = evaluate_at_offset(fixture, 0);
  auto const early = evaluate_at_offset(fixture, -30);
  auto const late = evaluate_at_offset(fixture, 30);

  ASSERT_TRUE(on_time.eligible());
  ASSERT_GE(on_time.predictions_.size(), 2U);
  EXPECT_NEAR(*on_time.delay_anchor_seconds_, 0, 2);
  EXPECT_EQ(on_time.transport_, fixture.run().t_);
  for (auto const& prediction : on_time.predictions_) {
    EXPECT_EQ(prediction.predicted_timestamp_seconds_ -
                  prediction.scheduled_timestamp_seconds_,
              prediction.delay_seconds_);
    EXPECT_EQ(prediction.delay_seconds_, *on_time.delay_anchor_seconds_);
  }
  ASSERT_TRUE(early.eligible());
  EXPECT_LT(*early.delay_anchor_seconds_, 0);
  ASSERT_TRUE(late.eligible());
  EXPECT_GT(*late.delay_anchor_seconds_, 0);
}

TEST(vehicle_prediction, dwell_anchors_delay_without_inventing_motion) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled() + 45;
  auto observations = std::vector<vehicle_observation>{
      fixture.observation(8.010, now - 10, 0.0, 20U, "STOPPED_AT"),
      fixture.observation(8.010, now, 0.0, 20U, "STOPPED_AT")};
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};

  auto const result = engine.evaluate(fixture.run(), observations, now);

  ASSERT_TRUE(result.eligible());
  EXPECT_EQ(result.delay_anchor_seconds_, 45);
  ASSERT_TRUE(result.confidence_.has_value());
  EXPECT_DOUBLE_EQ(result.confidence_->progress_velocity_mps_, 0.0);
}

TEST(vehicle_prediction, reported_speed_only_reduces_confidence_on_conflict) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled() - 50;
  auto observations = std::vector<vehicle_observation>{
      fixture.observation(8.004, now - 10, 40.0),
      fixture.observation(8.005, now, 40.0)};
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};

  auto const result = engine.evaluate(fixture.run(), observations, now);

  ASSERT_TRUE(result.eligible());
  ASSERT_TRUE(result.confidence_.has_value());
  EXPECT_TRUE(result.confidence_->reported_speed_conflict_);
  EXPECT_LT(result.confidence_->score_, 0.5);
}

TEST(vehicle_prediction, rejects_stale_insufficient_and_impossible_history) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled();
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};
  auto stale =
      std::vector<vehicle_observation>{fixture.observation(8.004, now - 401),
                                       fixture.observation(8.005, now - 400)};
  auto insufficient =
      std::vector<vehicle_observation>{fixture.observation(8.005, now)};
  auto impossible = std::vector<vehicle_observation>{
      fixture.observation(8.001, now - 1), fixture.observation(8.009, now)};

  EXPECT_EQ(engine.evaluate(fixture.run(), stale, now).diagnostics_.rejection_,
            vehicle_prediction_rejection_reason::kStaleHistory);
  EXPECT_EQ(
      engine.evaluate(fixture.run(), insufficient, now).diagnostics_.rejection_,
      vehicle_prediction_rejection_reason::kInsufficientHistory);
  EXPECT_EQ(
      engine.evaluate(fixture.run(), impossible, now).diagnostics_.rejection_,
      vehicle_prediction_rejection_reason::kImpossibleSpeed);
}

TEST(vehicle_prediction, rejects_terminal_and_handles_scheduled_short_turn) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled() + 300;
  auto at_terminal = std::vector<vehicle_observation>{
      fixture.observation(8.019, now - 10, 0.0, 30U),
      fixture.observation(8.020, now, 0.0, 30U, "STOPPED_AT")};
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};

  EXPECT_EQ(
      engine.evaluate(fixture.run(), at_terminal, now).diagnostics_.rejection_,
      vehicle_prediction_rejection_reason::kTerminal);

  auto short_turn = fixture.run();
  ++short_turn.stop_range_.from_;
  auto const short_now = fixture.next_scheduled() + 60;
  auto short_history = std::vector<vehicle_observation>{
      fixture.observation(8.011, short_now - 10, std::nullopt, 30U),
      fixture.observation(8.012, short_now, std::nullopt, 30U)};
  auto const result = engine.evaluate(short_turn, short_history, short_now);
  ASSERT_TRUE(result.eligible());
  EXPECT_EQ(result.transport_, short_turn.t_);
  EXPECT_EQ(result.delay_anchor_static_stop_sequence_, 30U);
}

TEST(vehicle_prediction, exports_only_high_confidence_stop_passages) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled() + 10;
  auto certain = std::vector<vehicle_observation>{
      fixture.observation(8.009, now - 10, std::nullopt, 20U),
      fixture.observation(8.011, now, std::nullopt, 30U)};
  auto engine = vehicle_prediction_engine{*fixture.data_->shapes_};

  auto const result = engine.evaluate(fixture.run(), certain, now);

  ASSERT_TRUE(result.eligible());
  ASSERT_EQ(result.observed_passages_.size(), 1U);
  EXPECT_EQ(result.observed_passages_.front().static_stop_sequence_, 20U);
  EXPECT_EQ(result.observed_passages_.front().uncertainty_seconds_, 10);
  EXPECT_EQ(result.diagnostics_.uncertain_passage_count_, 0U);
}

TEST(vehicle_prediction, excludes_uncertain_stop_passages_from_calibration) {
  auto fixture = prediction_fixture{};
  auto const now = fixture.next_scheduled() + 100;
  auto uncertain = std::vector<vehicle_observation>{
      fixture.observation(8.009, now - 100, std::nullopt, 20U),
      fixture.observation(8.011, now, std::nullopt, 30U)};
  auto engine = vehicle_prediction_engine{
      *fixture.data_->shapes_,
      vehicle_prediction_policy{.max_observation_age_seconds_ = 300,
                                .min_observations_ = 2U,
                                .min_progress_velocity_mps_ = 0.01,
                                .max_progress_velocity_mps_ = 55.0,
                                .max_predicted_travel_seconds_ = 4 * 60 * 60,
                                .max_passage_uncertainty_seconds_ = 90}};

  auto const result = engine.evaluate(fixture.run(), uncertain, now);

  ASSERT_TRUE(result.eligible());
  EXPECT_TRUE(result.observed_passages_.empty());
  EXPECT_EQ(result.diagnostics_.uncertain_passage_count_, 1U);
}

TEST(vehicle_prediction_cycle,
     rejects_route_only_unmatched_and_replacement_without_timing_changes) {
  auto fixture = prediction_fixture{};
  fixture.config_.timetable_->vehicle_eta_ = config::timetable::vehicle_eta{
      .mode_ = config::timetable::vehicle_eta::mode::shadow};
  auto const now = fixture.next_scheduled();
  auto position = [&](std::string entity, std::optional<std::string> trip,
                      std::optional<std::string> relationship = std::nullopt) {
    return vehicle_positions::vehicle_position{
        .feed_id_ = "prediction:vp",
        .entity_id_ = std::move(entity),
        .trip_ = {.trip_id_ = std::move(trip),
                  .start_date_ = "20260521",
                  .start_time_ = "01:00:00",
                  .route_id_ = "route",
                  .schedule_relationship_ = std::move(relationship)},
        .reported_position_ = {.pos_ = geo::latlng{50.0, 8.005}},
        .reported_time_ = now,
        .ingested_time_ = now};
  };
  auto positions = vehicle_positions::vehicle_position_store{};
  positions.replace_feed(
      "prediction:vp",
      {position("route-only", std::nullopt), position("unmatched", "unknown"),
       position("replacement", "straight", "REPLACEMENT")});
  auto const scheduled_before =
      fixture.run()[1].scheduled_time(n::event_type::kDep);

  auto const results = evaluate_vehicle_prediction_candidates(
      fixture.config_, *fixture.data_->tags_, *fixture.data_->tt_, nullptr,
      fixture.data_->shapes_.get(), positions, vehicle_observation_history{},
      now);

  ASSERT_EQ(results.size(), 3U);
  auto reasons = std::vector<vehicle_prediction_rejection_reason>{};
  for (auto const& result : results) {
    ASSERT_TRUE(result.batch_.diagnostics_.rejection_.has_value());
    reasons.push_back(*result.batch_.diagnostics_.rejection_);
  }
  EXPECT_NE(std::ranges::find(
                reasons, vehicle_prediction_rejection_reason::kMissingTripId),
            end(reasons));
  EXPECT_NE(std::ranges::find(
                reasons, vehicle_prediction_rejection_reason::kUnresolvedTrip),
            end(reasons));
  EXPECT_NE(
      std::ranges::find(
          reasons,
          vehicle_prediction_rejection_reason::kUnsupportedTripRelationship),
      end(reasons));
  EXPECT_EQ(scheduled_before,
            fixture.run()[1].scheduled_time(n::event_type::kDep));
}

}  // namespace
}  // namespace motis
