#pragma once

#include "osr/routing/route.h"

#include "nigiri/footpath_restriction.h"

namespace motis {

// Restriction record of an osr Pareto path (explicit unknowns, see
// osr::path_accessibility); fixed part = elevator rides.
inline nigiri::footpath_restriction to_restriction(osr::pareto_path const& p) {
  using fr = nigiri::footpath_restriction;
  auto const& a = p.accessibility_;
  auto r = fr{};
  if ((a.flags_ & osr::path_accessibility::kSteps) != 0U) {
    r.flags_ |= fr::kStairs;
  }
  if ((a.flags_ & osr::path_accessibility::kElevator) != 0U) {
    r.flags_ |= fr::kElevator;
  }
  if ((a.flags_ & osr::path_accessibility::kRamp) != 0U) {
    r.flags_ |= fr::kRamp;
  }
  r.slope_ =
      static_cast<std::uint8_t>(fr::to_slope(a.max_incline_ / 10.0) |
                                (a.has_unknown_incline_ ? fr::kUnknown : 0U));
  r.width_ = static_cast<std::uint8_t>(
      (a.has_width() ? fr::to_width(a.min_width_cm_ / 100.0) : fr::kNoWidth) |
      (a.has_unknown_width_ ? fr::kUnknown : 0U));
  r.fixed_ = fr::to_fixed(static_cast<unsigned>(p.fixed_.count()));
  return r;
}

// Footpath minutes of an osr Pareto path (rounded up after summing).
inline nigiri::duration_t to_minutes(osr::pareto_path const& p) {
  return nigiri::duration_t{
      static_cast<nigiri::duration_t::rep>((p.duration().count() + 59) / 60)};
}

}  // namespace motis
