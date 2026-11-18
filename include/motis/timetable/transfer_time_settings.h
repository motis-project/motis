#pragma once

#include <bit>
#include <cstdint>
#include <utility>

#include "utl/verify.h"

#include "net/bad_request_exception.h"

#include "motis-api/motis-api.h"

#include "nigiri/footpath_restriction.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace motis {

template <typename Query>
bool use_routed_transfers(Query const& query) {
  return query.useRoutedTransfers_ || query.transferTimeFactor_ != 1.0 ||
         query.additionalTransferTime_ != 0;
}

inline nigiri::profile_idx_t get_profile_idx(
    bool const use_routed_transfers,
    bool const require_car_transport,
    api::PedestrianProfileEnum const pedestrian_profile) {
  if (!use_routed_transfers) {
    return nigiri::kDefaultProfile;
  }
  if (require_car_transport) {
    return nigiri::kCarProfile;
  }
  return pedestrian_profile == api::PedestrianProfileEnum::WHEELCHAIR
             ? nigiri::kWheelchairProfile
             : nigiri::kFootProfile;
}

template <typename Query>
nigiri::profile_idx_t get_profile_idx(Query const& query) {
  return get_profile_idx(use_routed_transfers(query),
                         query.requireCarTransport_, query.pedestrianProfile_);
}

// Wheelchair queries keep the wheelchair profile (trip / stop wheelchair
// filters); its footpaths are the foot lists (one list, plan §0.3, see
// timetable::footpath_profile): the wheelchair restrictions exclude steps,
// get_transfer_time_settings(tt, query) slows the walking part down.
inline nigiri::profile_idx_t get_available_profile(
    nigiri::timetable const& tt, nigiri::profile_idx_t const prf_idx) {
  return tt.locations_.footpaths_out_
                 .at(tt.locations_.footpath_profile(prf_idx))
                 .empty()
             ? nigiri::kDefaultProfile
             : prf_idx;
}

// Walking speed foot / wheelchair (osr: 1.2 m/s vs. 0.8 m/s).
constexpr auto const kWheelchairWalkFactor = 1.5F;

template <typename Query>
nigiri::profile_idx_t get_profile_idx(nigiri::timetable const& tt,
                                      Query const& query) {
  return get_available_profile(tt, get_profile_idx(query));
}

inline std::uint8_t to_restriction_flag(api::WalkFeatureEnum const f) {
  using fr = nigiri::footpath_restriction;
  switch (f) {
    case api::WalkFeatureEnum::STAIRS: return fr::kStairs;
    case api::WalkFeatureEnum::ESCALATOR: return fr::kEscalator;
    case api::WalkFeatureEnum::ELEVATOR: return fr::kElevator;
    case api::WalkFeatureEnum::RAMP: return fr::kRamp;
    case api::WalkFeatureEnum::STEP: return fr::kStep;
  }
  std::unreachable();
}

// Bit position of a path kind flag = index into the penalty array.
inline unsigned to_penalty_idx(std::uint8_t const flag) {
  return static_cast<unsigned>(std::countr_zero(flag));
}

// Penalties of the presets for single steps (minutes). Assumed values, not
// from the DELFI handbook: a step slows a walker down and needs a helper to
// lift an assisted wheelchair.
constexpr auto const kWalkerStepPenalty = std::uint8_t{1U};
constexpr auto const kAssistedStepPenalty = std::uint8_t{2U};

// DELFI handbook on barrier-free travel information (Anlage 3): all groups
// need >= 90 cm and no stairs / escalators. Single steps (<= 15 cm, the
// height is not modelled) are fine with a walker or assistance.
inline nigiri::routing::footpath_restriction_settings get_preset(
    api::AccessibilityProfileEnum const p) {
  using fr = nigiri::footpath_restriction;
  constexpr auto const kNoStairs =
      static_cast<std::uint8_t>(fr::kStairs | fr::kEscalator);
  auto const preset = [](unsigned const forbidden, double const slope,
                         std::uint8_t const step_penalty) {
    auto s = nigiri::routing::footpath_restriction_settings{
        .forbidden_ = static_cast<std::uint8_t>(forbidden),
        .max_slope_ = fr::to_slope(slope),
        .min_width_ = fr::to_width(0.9)};
    s.penalty_[to_penalty_idx(fr::kStep)] = step_penalty;
    return s;
  };
  switch (p) {
    case api::AccessibilityProfileEnum::NONE: return {};
    case api::AccessibilityProfileEnum::WHEELCHAIR:
      return preset(kNoStairs | fr::kStep | fr::kBanWheelchair, 6.0, 0U);
    case api::AccessibilityProfileEnum::WHEELCHAIR_ASSISTED:
      return preset(kNoStairs | fr::kBanWheelchairAssisted, 12.0,
                    kAssistedStepPenalty);
    case api::AccessibilityProfileEnum::WALKER:
      return preset(kNoStairs | fr::kBanWalker, 6.0, kWalkerStepPenalty);
  }
  std::unreachable();
}

template <typename Query>
nigiri::routing::footpath_restriction_settings get_restriction_settings(
    Query const& query) {
  using fr = nigiri::footpath_restriction;
  auto s = get_preset(
      query.accessibilityProfile_ == api::AccessibilityProfileEnum::NONE &&
              query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
          ? api::AccessibilityProfileEnum::WHEELCHAIR
          : query.accessibilityProfile_);
  if (query.avoidWalkFeatures_.has_value()) {
    for (auto const f : *query.avoidWalkFeatures_) {
      s.forbidden_ |= to_restriction_flag(f);
    }
  }
  if (query.penalizedWalkFeatures_.has_value()) {
    utl::verify<net::bad_request_exception>(
        query.walkFeaturePenalty_ >= 0 && query.walkFeaturePenalty_ <= 60,
        "walkFeaturePenalty={} must be in [0, 60]", query.walkFeaturePenalty_);
    for (auto const f : *query.penalizedWalkFeatures_) {
      s.penalty_[to_penalty_idx(to_restriction_flag(f))] =
          static_cast<std::uint8_t>(query.walkFeaturePenalty_);
    }
  }
  if (query.maxSlope_.has_value()) {
    utl::verify<net::bad_request_exception>(*query.maxSlope_ >= 0.0,
                                            "maxSlope={} must not be negative",
                                            *query.maxSlope_);
    s.max_slope_ = fr::to_slope(*query.maxSlope_);
  }
  if (query.minWidth_.has_value()) {
    utl::verify<net::bad_request_exception>(*query.minWidth_ >= 0.0,
                                            "minWidth={} must not be negative",
                                            *query.minWidth_);
    s.min_width_ = fr::to_width(*query.minWidth_);
  }
  s.strict_ = query.strictAccessibility_;
  return s;
}

template <typename Query>
nigiri::routing::transfer_time_settings get_transfer_time_settings(
    Query const& query) {
  utl::verify<net::bad_request_exception>(
      query.minTransferTime_ >= 0, "minTransferTime={} must not be negative",
      query.minTransferTime_);
  utl::verify<net::bad_request_exception>(
      query.additionalTransferTime_ >= 0,
      "additionalTransferTime={} must not be negative",
      query.additionalTransferTime_);
  utl::verify<net::bad_request_exception>(
      query.transferTimeFactor_ >= 1.0,
      "transferTimeFactor={} must not be smaller than 1",
      query.transferTimeFactor_);
  return {.default_ = (query.minTransferTime_ == 0 &&
                       query.additionalTransferTime_ == 0 &&
                       query.transferTimeFactor_ == 1.0),
          .min_transfer_time_ = nigiri::duration_t{query.minTransferTime_},
          .additional_time_ = nigiri::duration_t{query.additionalTransferTime_},
          .factor_ = static_cast<float>(query.transferTimeFactor_),
          .restrictions_ = get_restriction_settings(query)};
}

template <typename Query>
nigiri::routing::transfer_time_settings get_transfer_time_settings(
    nigiri::timetable const& tt, Query const& query) {
  auto s = get_transfer_time_settings(query);
  if (get_profile_idx(tt, query) == nigiri::kWheelchairProfile &&
      tt.locations_.footpath_profile(nigiri::kWheelchairProfile) ==
          nigiri::kFootProfile) {
    s.default_ = false;
    s.factor_ *= kWheelchairWalkFactor;
  }
  return s;
}

}  // namespace motis
