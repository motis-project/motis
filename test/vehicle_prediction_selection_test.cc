#include "gtest/gtest.h"

#include <optional>
#include <string>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/rt/vehicle_prediction_selection.h"

namespace n = nigiri;

namespace motis {
namespace {

constexpr auto kNow = std::int64_t{10'000};

n::transport transport(unsigned const index = 1U, unsigned const day = 2U) {
  return {n::transport_idx_t{index}, n::day_idx_t{day}};
}

vehicle_prediction_selector_policy policy() {
  return {.max_candidate_age_seconds_ = 120,
          .max_timestamp_skew_seconds_ = 10,
          .max_progress_difference_m_ = 100.0,
          .min_provider_confidence_ = 0.5,
          .min_gps_confidence_ = 0.6,
          .min_source_switch_confidence_advantage_ = 0.1,
          .state_ttl_seconds_ = 300,
          .flap_window_seconds_ = 120,
          .minute_boundary_hysteresis_seconds_ = 5};
}

timing_source_candidate candidate(vehicle_prediction_source const source,
                                  double const confidence,
                                  double const progress,
                                  std::int64_t const timestamp = kNow) {
  return {.source_ = source,
          .reference_timestamp_seconds_ = timestamp,
          .confidence_ = confidence,
          .physically_reachable_ = true,
          .implied_progress_m_ = progress,
          .predictions_ = {{.static_stop_sequence_ = 20U,
                            .scheduled_timestamp_seconds_ = 11'000,
                            .predicted_timestamp_seconds_ = 11'060,
                            .delay_seconds_ = 60,
                            .horizon_seconds_ = 1'060}}};
}

vehicle_prediction_selection_input both(double const provider_confidence = 0.9,
                                        double const gps_confidence = 0.8,
                                        double const provider_progress = 1'000,
                                        double const gps_progress = 1'010,
                                        std::int64_t const now = kNow) {
  return {.transport_ = transport(),
          .now_seconds_ = now,
          .provider_ = candidate(vehicle_prediction_source::kProvider,
                                 provider_confidence, provider_progress, now),
          .gps_ = candidate(vehicle_prediction_source::kGps, gps_confidence,
                            gps_progress, now)};
}

TEST(provider_timing_extraction,
     reads_timing_without_mutating_operational_semantics) {
  auto message = transit_realtime::FeedMessage{};
  message.mutable_header()->set_gtfs_realtime_version("2.0");
  message.mutable_header()->set_timestamp(kNow);
  auto* const entity = message.add_entity();
  entity->set_id("timed");
  auto* const update = entity->mutable_trip_update();
  update->mutable_trip()->set_trip_id("trip");
  update->set_timestamp(kNow - 1);
  auto* const timed = update->add_stop_time_update();
  timed->set_stop_sequence(20U);
  timed->mutable_arrival()->set_delay(75);
  timed->set_schedule_relationship(
      transit_realtime::TripUpdate_StopTimeUpdate::SKIPPED);
  timed->mutable_stop_time_properties()->set_assigned_stop_id("platform-b");
  timed->mutable_stop_time_properties()->set_stop_headsign("short turn");
  auto* const no_data = update->add_stop_time_update();
  no_data->set_stop_sequence(30U);
  no_data->set_schedule_relationship(
      transit_realtime::TripUpdate_StopTimeUpdate::NO_DATA);
  no_data->mutable_departure()->set_delay(999);

  auto* const operational = message.add_entity();
  operational->set_id("cancelled");
  operational->mutable_trip_update()->mutable_trip()->set_trip_id("trip");
  operational->mutable_trip_update()->mutable_trip()->set_schedule_relationship(
      transit_realtime::TripDescriptor::CANCELED);
  auto const before = message.SerializeAsString();

  auto const extracted = extract_provider_timing(
      message, [](transit_realtime::TripDescriptor const&) {
        return std::optional{resolved_provider_trip{
            .transport_ = transport(),
            .stops_ = {{.static_stop_sequence_ = 20U,
                        .arrival_timestamp_seconds_ = 20'000,
                        .departure_timestamp_seconds_ = 20'010},
                       {.static_stop_sequence_ = 30U,
                        .arrival_timestamp_seconds_ = 21'000,
                        .departure_timestamp_seconds_ = 21'010}}}};
      });

  ASSERT_EQ(extracted.candidates_.size(), 1U);
  ASSERT_EQ(extracted.candidates_.front().stops_.size(), 1U);
  EXPECT_EQ(extracted.candidates_.front().transport_, transport());
  EXPECT_EQ(
      extracted.candidates_.front().stops_.front().arrival_timestamp_seconds_,
      20'075);
  EXPECT_EQ(extracted.operational_only_trip_updates_, 1U);
  EXPECT_EQ(message.SerializeAsString(), before);
  EXPECT_EQ(message.entity(0)
                .trip_update()
                .stop_time_update(0)
                .schedule_relationship(),
            transit_realtime::TripUpdate_StopTimeUpdate::SKIPPED);
  EXPECT_EQ(message.entity(0)
                .trip_update()
                .stop_time_update(0)
                .stop_time_properties()
                .assigned_stop_id(),
            "platform-b");
}

TEST(provider_timing_extraction, missing_trip_update_is_not_on_time) {
  auto message = transit_realtime::FeedMessage{};
  message.mutable_header()->set_gtfs_realtime_version("2.0");
  message.add_entity()->set_id("vehicle-only");
  auto const extracted = extract_provider_timing(
      message, [](transit_realtime::TripDescriptor const&) {
        ADD_FAILURE() << "resolver must not run without a TripUpdate";
        return std::optional<resolved_provider_trip>{};
      });
  EXPECT_TRUE(extracted.candidates_.empty());
  EXPECT_EQ(extracted.operational_only_trip_updates_, 0U);
}

TEST(vehicle_prediction_selector, handles_each_candidate_availability_case) {
  auto input = both();
  EXPECT_EQ(select_vehicle_prediction_source(input, policy()).source_,
            vehicle_prediction_source::kProvider);

  input.provider_.reset();
  EXPECT_EQ(select_vehicle_prediction_source(input, policy()).source_,
            vehicle_prediction_source::kGps);
  input.gps_.reset();
  EXPECT_EQ(select_vehicle_prediction_source(input, policy()).source_,
            vehicle_prediction_source::kSchedule);

  input.provider_ = candidate(vehicle_prediction_source::kProvider, 0.9, 1000);
  EXPECT_EQ(select_vehicle_prediction_source(input, policy()).source_,
            vehicle_prediction_source::kProvider);
}

TEST(vehicle_prediction_selector,
     rejects_unreachable_stale_and_uncalibrated_candidates) {
  auto input = both();
  input.provider_->physically_reachable_ = false;
  input.gps_->reference_timestamp_seconds_ = kNow - 121;
  auto selection = select_vehicle_prediction_source(input, policy());
  EXPECT_EQ(selection.source_, vehicle_prediction_source::kSchedule);
  EXPECT_EQ(selection.diagnostics_.provider_rejection_,
            timing_candidate_rejection_reason::kPhysicallyUnreachable);
  EXPECT_EQ(selection.diagnostics_.gps_rejection_,
            timing_candidate_rejection_reason::kStale);

  auto unavailable_policy = policy();
  unavailable_policy.max_progress_difference_m_ = -1.0;
  selection = select_vehicle_prediction_source(both(), unavailable_policy);
  EXPECT_EQ(selection.source_, vehicle_prediction_source::kSchedule);
  EXPECT_EQ(selection.reason_,
            vehicle_prediction_selection_reason::kPolicyUnavailable);
}

TEST(vehicle_prediction_selector,
     rejects_provider_when_timestamps_or_progress_are_incomparable) {
  auto input = both();
  input.provider_->reference_timestamp_seconds_ -= 11;
  auto selection = select_vehicle_prediction_source(input, policy());
  EXPECT_EQ(selection.source_, vehicle_prediction_source::kGps);
  EXPECT_EQ(selection.diagnostics_.provider_rejection_,
            timing_candidate_rejection_reason::kTimestampNotComparable);

  input = both();
  input.provider_->implied_progress_m_.reset();
  selection = select_vehicle_prediction_source(input, policy());
  EXPECT_EQ(selection.source_, vehicle_prediction_source::kGps);
  EXPECT_EQ(selection.diagnostics_.provider_rejection_,
            timing_candidate_rejection_reason::kProgressNotComparable);

  input = both(0.9, 0.8, 1'000, 1'101);
  selection = select_vehicle_prediction_source(input, policy());
  EXPECT_EQ(selection.source_, vehicle_prediction_source::kGps);
  EXPECT_EQ(selection.reason_,
            vehicle_prediction_selection_reason::kProviderProgressInconsistent);
}

TEST(vehicle_prediction_selection_state,
     provider_requires_two_consistent_cycles_to_recover) {
  auto state = vehicle_prediction_selection_state{};
  auto first = state.select(both(), policy());
  EXPECT_EQ(first.source_, vehicle_prediction_source::kProvider);

  auto inconsistent = both(0.9, 0.8, 1'000, 1'101, kNow + 10);
  auto rejected = state.select(inconsistent, policy());
  EXPECT_EQ(rejected.source_, vehicle_prediction_source::kGps);
  EXPECT_TRUE(rejected.diagnostics_.source_transition_);

  auto one = state.select(both(0.9, 0.8, 1'000, 1'010, kNow + 20), policy());
  EXPECT_EQ(one.source_, vehicle_prediction_source::kGps);
  EXPECT_EQ(one.reason_,
            vehicle_prediction_selection_reason::kProviderRecoveryPending);
  EXPECT_EQ(one.diagnostics_.provider_consistent_cycles_, 1U);

  auto two = state.select(both(0.9, 0.8, 1'000, 1'010, kNow + 30), policy());
  EXPECT_EQ(two.source_, vehicle_prediction_source::kProvider);
  EXPECT_EQ(two.reason_,
            vehicle_prediction_selection_reason::kProviderRecovered);
  EXPECT_TRUE(two.diagnostics_.provider_recovery_);
  EXPECT_TRUE(two.diagnostics_.source_transition_);
  EXPECT_TRUE(two.diagnostics_.flap_);
}

TEST(vehicle_prediction_selection_state,
     confidence_hysteresis_prevents_threshold_flapping) {
  auto state = vehicle_prediction_selection_state{};
  EXPECT_EQ(state.select(both(0.8, 0.79), policy()).source_,
            vehicle_prediction_source::kProvider);
  auto slight_gps_advantage =
      state.select(both(0.8, 0.85, 1'000, 1'010, kNow + 10), policy());
  EXPECT_EQ(slight_gps_advantage.source_, vehicle_prediction_source::kProvider);
  EXPECT_EQ(slight_gps_advantage.reason_,
            vehicle_prediction_selection_reason::kSourceHysteresis);
}

TEST(vehicle_prediction_selection_state,
     expires_inactive_and_completed_state_across_snapshot_styles) {
  auto state = vehicle_prediction_selection_state{};
  (void)state.select(both(), policy());  // full replacement cycle
  EXPECT_EQ(state.size(), 1U);
  (void)state.select(both(0.9, 0.8, 1'000, 1'010, kNow + 60),
                     policy());  // incremental cycle
  EXPECT_EQ(state.size(), 1U);
  state.expire(kNow + 361, policy());
  EXPECT_EQ(state.size(), 0U);
  (void)state.select(both(0.9, 0.8, 1'000, 1'010, kNow + 400), policy());
  EXPECT_EQ(state.size(), 1U);
  (void)state.select(both(0.9, 0.8, 1'000, 1'010, kNow + 410), policy(), true);
  EXPECT_EQ(state.size(), 0U);
}

TEST(vehicle_prediction_minute_rendering,
     uses_cpp_truncation_for_positive_and_negative_seconds) {
  EXPECT_EQ(rendered_delay_minutes(119), 1);
  EXPECT_EQ(rendered_delay_minutes(59), 0);
  EXPECT_EQ(rendered_delay_minutes(-59), 0);
  EXPECT_EQ(rendered_delay_minutes(-119), -1);
}

TEST(vehicle_prediction_minute_rendering,
     hysteresis_prevents_boundary_oscillation_without_changing_raw_seconds) {
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(62, 0, 5), 0);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(65, 0, 5), 1);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(58, 1, 5), 1);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(55, 1, 5), 0);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(-62, 0, 5), 0);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(-65, 0, 5), -1);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(-58, -1, 5), -1);
  EXPECT_EQ(rendered_delay_minutes_with_hysteresis(-55, -1, 5), 0);
}

TEST(vehicle_prediction_minute_rendering,
     selection_state_keeps_raw_seconds_while_stabilizing_minutes) {
  auto state = vehicle_prediction_selection_state{};
  auto input = both();
  input.provider_->predictions_.front().delay_seconds_ = 65;
  auto selected = state.select(input, policy());
  ASSERT_EQ(selected.diagnostics_.rendered_delays_.size(), 1U);
  EXPECT_EQ(selected.diagnostics_.rendered_delays_.front().raw_delay_seconds_,
            65);
  EXPECT_EQ(
      selected.diagnostics_.rendered_delays_.front().rendered_delay_minutes_,
      1);

  input.now_seconds_ += 10;
  input.provider_->reference_timestamp_seconds_ += 10;
  input.gps_->reference_timestamp_seconds_ += 10;
  input.provider_->predictions_.front().delay_seconds_ = 58;
  selected = state.select(input, policy());
  EXPECT_EQ(selected.source_, vehicle_prediction_source::kProvider);
  EXPECT_EQ(selected.diagnostics_.rendered_delays_.front().raw_delay_seconds_,
            58);
  EXPECT_EQ(
      selected.diagnostics_.rendered_delays_.front().rendered_delay_minutes_,
      1);
}

TEST(vehicle_prediction_shadow_cycle,
     resolves_feed_mode_policy_and_aggregates_without_effective_changes) {
  auto c = config{};
  c.timetable_.emplace();
  c.timetable_->vehicle_eta_ = config::timetable::vehicle_eta{
      .mode_ = config::timetable::vehicle_eta::mode::shadow,
      .feeds_ = {
          {"A", config::timetable::vehicle_eta::feed{
                    .modes_ = std::vector<std::string>{"BUS"},
                    .mode_ = config::timetable::vehicle_eta::mode::off}}}};
  auto requests = std::vector<vehicle_prediction_shadow_request>{
      {.feed_ = "A", .mode_ = n::clasz::kBus, .selection_ = both()},
      {.feed_ = "A", .mode_ = n::clasz::kTram, .selection_ = both()},
      {.feed_ = "B",
       .mode_ = n::clasz::kBus,
       .selection_ = vehicle_prediction_selection_input{
           .transport_ = transport(2U), .now_seconds_ = kNow}}};
  auto state = vehicle_prediction_selection_state{};
  auto const result = evaluate_vehicle_prediction_shadow_cycle(
      c, requests,
      [](std::string_view const feed, n::clasz const mode)
          -> std::optional<vehicle_prediction_selector_policy> {
        EXPECT_TRUE((feed == "A" && mode == n::clasz::kTram) ||
                    (feed == "B" && mode == n::clasz::kBus));
        return policy();
      },
      state);

  ASSERT_EQ(result.selections_.size(), 2U);
  EXPECT_EQ(result.selections_[0].source_,
            vehicle_prediction_source::kProvider);
  EXPECT_EQ(result.selections_[1].source_,
            vehicle_prediction_source::kSchedule);
  ASSERT_EQ(result.summaries_.size(), 2U);
  EXPECT_EQ(result.summaries_[0].provider_selected_, 1U);
  EXPECT_EQ(result.summaries_[1].schedule_fallback_, 1U);
  EXPECT_EQ(result.summaries_[1].flap_rate(), 0.0);
}

}  // namespace
}  // namespace motis
