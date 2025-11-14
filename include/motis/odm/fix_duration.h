#pragma once

#include "nigiri/types.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/journey.h"

#include  "utl/erase_if.h"

#include "motis/transport_mode_ids.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima.h"

namespace motis::odm {

template <typename A, typename B>
void fix_first_mile_duration(std::vector<nigiri::routing::journey>& journeys,
                             std::vector<A> const& first_mile,
                             std::vector<B> const& prev_first_mile,
                             nigiri::transport_mode_id_t const mode) {
  for (auto const [curr, prev] : utl::zip(first_mile, prev_first_mile)) {

    auto const uses_prev = [&,
                            prev2 = prev /* hack for MacOS - fixed with 16 */](
                               nigiri::routing::journey const& j) {
      return j.legs_.size() > 1 &&
             j.legs_.front().dep_time_ == prev2.time_at_start_ &&
             j.legs_.front().arr_time_ >= prev2.time_at_stop_ &&
             (j.legs_.front().arr_time_ == prev2.time_at_stop_ ||
              mode == kRideSharingTransportModeId) &&
             j.legs_.front().to_ == prev2.stop_ &&
             is_odm_leg(j.legs_.front(), mode);
    };

    if (curr.time_at_start_ == kInfeasible) {
      utl::erase_if(journeys, uses_prev);
    } else {
      for (auto& j : journeys) {
        if (uses_prev(j)) {
          auto const l = begin(j.legs_);
          if (std::holds_alternative<nigiri::footpath>(std::next(l)->uses_)) {
            continue;  // odm leg fixed already before with a different
                       // time_at_stop (rideshare)
          }
          l->dep_time_ = curr.time_at_start_;
          l->arr_time_ =
              curr.time_at_stop_ - (mode == kRideSharingTransportModeId
                                        ? kODMTransferBuffer
                                        : nigiri::duration_t{0});
          std::get<nigiri::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              std::next(l), nigiri::direction::kForward, l->to_, l->to_,
              l->arr_time_, std::next(l)->dep_time_,
              nigiri::footpath{l->to_, std::next(l)->dep_time_ - l->arr_time_});
        }
      }
    }
  }
}

template <typename A, typename B>
void fix_last_mile_duration(std::vector<nigiri::routing::journey>& journeys,
                            std::vector<A> const& last_mile,
                            std::vector<B> const& prev_last_mile,
                            nigiri::transport_mode_id_t const mode) {
  for (auto const [curr, prev] : utl::zip(last_mile, prev_last_mile)) {
    auto const uses_prev =
        [&, prev2 = prev /* hack for MacOS - fixed with 16 */](auto const& j) {
          return j.legs_.size() > 1 &&
                 j.legs_.back().dep_time_ <= prev2.time_at_stop_ &&
                 (j.legs_.back().dep_time_ == prev2.time_at_stop_ ||
                  mode == kRideSharingTransportModeId) &&
                 j.legs_.back().arr_time_ == prev2.time_at_start_ &&
                 j.legs_.back().from_ == prev2.stop_ &&
                 is_odm_leg(j.legs_.back(), mode);
        };

    if (curr.time_at_start_ == kInfeasible) {
      utl::erase_if(journeys, uses_prev);
    } else {
      for (auto& j : journeys) {
        if (uses_prev(j)) {
          auto const l = std::prev(end(j.legs_));
          if (std::holds_alternative<nigiri::footpath>(std::prev(l)->uses_)) {
            continue;  // odm leg fixed already before with a different
                       // time_at_stop (rideshare)
          }
          l->dep_time_ =
              curr.time_at_stop_ + (mode == kRideSharingTransportModeId
                                        ? kODMTransferBuffer
                                        : nigiri::duration_t{0});
          l->arr_time_ = curr.time_at_start_;
          std::get<nigiri::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              l, nigiri::direction::kForward, l->from_, l->from_,
              std::prev(l)->arr_time_, l->dep_time_,
              nigiri::footpath{l->from_, l->dep_time_ - std::prev(l)->arr_time_});
        }
      }
    }
  }
}

} // namespace motis::odm