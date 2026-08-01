#include "motis/rt/vehicle_prediction_diagnostics.h"

#include <chrono>
#include <optional>
#include <string_view>

#include "nigiri/rt/frun.h"

#include "motis/config.h"
#include "motis/rt/vehicle_matching.h"
#include "motis/rt/vehicle_observation_history.h"
#include "motis/rt/vehicle_position.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

namespace n = nigiri;

namespace motis {
namespace {

std::string_view feed_tag(std::string_view const feed) {
  auto const separator = feed.find(':');
  return separator == std::string_view::npos ? feed : feed.substr(0, separator);
}

vehicle_key key_for(vehicle_positions::vehicle_position const& vehicle) {
  return vehicle.vehicle_.id_.has_value()
             ? vehicle_key{vehicle.feed_id_, *vehicle.vehicle_.id_,
                           vehicle_key_source::kVehicleDescriptor}
             : vehicle_key{vehicle.feed_id_, vehicle.entity_id_,
                           vehicle_key_source::kEntityId};
}

vehicle_prediction_cycle_result rejected(
    std::string feed, vehicle_prediction_rejection_reason const reason) {
  auto result = vehicle_prediction_cycle_result{.feed_ = std::move(feed)};
  result.batch_.diagnostics_.rejection_ = reason;
  return result;
}

}  // namespace

std::vector<vehicle_prediction_cycle_result>
evaluate_vehicle_prediction_candidates(
    config const& c,
    tag_lookup const& tags,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::shapes_storage const* shapes,
    vehicle_positions::vehicle_position_store const& positions,
    vehicle_observation_history const& history,
    std::int64_t const now) {
  auto results = std::vector<vehicle_prediction_cycle_result>{};
  if (!c.timetable_ || !c.timetable_->vehicle_eta_ || shapes == nullptr) {
    return results;
  }
  auto engine = vehicle_prediction_engine{
      *shapes, vehicle_prediction_policy{
                   .max_observation_age_seconds_ =
                       c.timetable_->vehicle_eta_->history_.max_age_seconds_}};
  for (auto const& position : positions.all()) {
    auto feed = std::string{feed_tag(position.feed_id_)};
    if (!position.trip_.trip_id_.has_value()) {
      results.push_back(
          rejected(std::move(feed),
                   vehicle_prediction_rejection_reason::kMissingTripId));
      continue;
    }
    if (position.trip_.schedule_relationship_.has_value() &&
        *position.trip_.schedule_relationship_ != "SCHEDULED") {
      results.push_back(rejected(
          std::move(feed),
          vehicle_prediction_rejection_reason::kUnsupportedTripRelationship));
      continue;
    }
    auto run = vehicle_matching::resolve_run(tags, tt, rtt, position);
    if (!run.has_value()) {
      results.push_back(
          rejected(std::move(feed),
                   vehicle_prediction_rejection_reason::kUnresolvedTrip));
      continue;
    }
    if (!run->is_scheduled()) {
      results.push_back(
          rejected(std::move(feed),
                   vehicle_prediction_rejection_reason::kUnscheduledTrip));
      continue;
    }
    auto const mode = (*run)[0].get_clasz(n::event_type::kDep);
    if (c.vehicle_eta_mode(feed, mode) ==
        config::timetable::vehicle_eta::mode::off) {
      continue;
    }
    auto result = vehicle_prediction_cycle_result{.feed_ = std::move(feed),
                                                  .trip_id_ = tags.id(
                                                      tt, (*run)[0],
                                                      n::event_type::kDep),
                                                  .mode_ = mode};
    result.batch_ =
        engine.evaluate(*run, history.observations(key_for(position)), now);
    if (result.batch_.eligible()) {
      for (auto const& prediction : result.batch_.predictions_) {
        auto const stop_count = run->stop_range_.size();
        for (auto i = n::stop_idx_t{0U}; i != stop_count; ++i) {
          auto const event =
              i == 0U ? n::event_type::kDep : n::event_type::kArr;
          auto const scheduled = to_seconds((*run)[i].scheduled_time(event));
          if (scheduled != prediction.scheduled_timestamp_seconds_) {
            continue;
          }
          auto const provider = to_seconds((*run)[i].time(event));
          if (!run->is_rt()) {
            break;
          }
          result.provider_predictions_.push_back(
              {.static_stop_sequence_ = prediction.static_stop_sequence_,
               .scheduled_timestamp_seconds_ = scheduled,
               .predicted_timestamp_seconds_ = provider,
               .delay_seconds_ = provider - scheduled,
               .horizon_seconds_ = provider - now});
          auto const raw = prediction.predicted_timestamp_seconds_ - provider;
          result.provider_raw_error_seconds_.push_back(raw);
          result.provider_minute_error_.push_back(
              prediction.predicted_timestamp_seconds_ / 60 - provider / 60);
          break;
        }
      }
    }
    results.emplace_back(std::move(result));
  }
  return results;
}

}  // namespace motis
