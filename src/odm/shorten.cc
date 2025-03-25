#include "motis/odm/odm.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"

namespace motis::odm {

namespace n = nigiri;
namespace nr = nigiri::routing;

void shorten(std::vector<nr::journey>& odm_journeys,
             std::vector<nr::start> const& from_rides,
             std::vector<nr::start> const& to_rides,
             n::timetable const& tt,
             n::rt_timetable const* rtt,
             api::plan_params const& query) {

  auto const shorten_first_leg = [&](n::routing::journey& j) {
    auto& odm_leg = begin(j.legs_)[0];
    auto& pt_leg = begin(j.legs_)[1];

    if (!is_odm_leg(odm_leg) ||
        !std::holds_alternative<nr::journey::run_enter_exit>(pt_leg.uses_)) {
      return;
    }

    auto const& ree = std::get<nr::journey::run_enter_exit>(pt_leg.uses_);
    auto run = n::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.to_ = ree.stop_range_.to_;
    auto min_stop_idx = ree.stop_range_.from_;
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<n::routing::start>{};
    for (auto const stop : run) {
      if (stop.is_canceled() ||
          !stop.in_allowed(query.pedestrianProfile_ ==
                           api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kDep))) {
        continue;
      }
      for (auto& ride : from_rides) {
        if (n::routing::matches(tt,
                                n::routing::location_match_mode::kEquivalent,
                                ride.stop_, stop.get_location_idx()) &&
            ride.time_at_stop_ == stop.time(n::event_type::kDep)) {
          auto const cur_odm_duration = duration(ride);
          if (cur_odm_duration < min_odm_duration) {
            min_stop_idx = stop.stop_idx_;
            min_odm_duration = cur_odm_duration;
            shorter_ride = ride;
          }
          break;
        }
      }
    }
    if (shorter_ride) {
      std::println("Found a shorter ODM first leg: [{},{}]",
                   tt.locations_.get(shorter_ride->stop_).name_,
                   min_odm_duration);
      auto& odm_offset = std::get<n::routing::offset>(odm_leg.uses_);
      auto& pt_run_enter_exit =
          std::get<n::routing::journey::run_enter_exit>(pt_leg.uses_);
      j.start_time_ = odm_leg.dep_time_ = shorter_ride->time_at_start_;
      odm_offset.duration_ = min_odm_duration;
      odm_leg.arr_time_ = pt_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_leg.to_ = odm_offset.target_ = pt_leg.from_ = shorter_ride->stop_;
      pt_run_enter_exit.stop_range_.from_ =
          pt_run_enter_exit.r_.stop_range_.from_ = min_stop_idx;
    }
  };

  auto const shorten_last_leg = [&](n::routing::journey& j) {
    auto& odm_leg = rbegin(j.legs_)[0];
    auto& pt_leg = rbegin(j.legs_)[1];

    if (!is_odm_leg(odm_leg) ||
        !std::holds_alternative<n::routing::journey::run_enter_exit>(
            pt_leg.uses_)) {
      return;
    }

    auto const& ree =
        std::get<n::routing::journey::run_enter_exit>(pt_leg.uses_);
    auto run = nigiri::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.from_ = ree.stop_range_.from_ + 1U;
    auto min_stop_idx = static_cast<n::stop_idx_t>(ree.stop_range_.to_ - 1U);
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<n::routing::start>{};
    std::println("\nmin_start: {},{}", tt.locations_.get(odm_leg.from_).name_,
                 min_odm_duration);
    for (auto const stop : run) {
      std::println("examining stop: {}", stop.name());
      if (stop.is_canceled() ||
          !stop.out_allowed(query.pedestrianProfile_ ==
                            api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kArr))) {
        continue;
      }
      for (auto& ride : to_rides) {
        if (n::routing::matches(tt,
                                n::routing::location_match_mode::kEquivalent,
                                ride.stop_, stop.get_location_idx()) &&
            ride.time_at_stop_ == stop.time(n::event_type::kArr)) {
          auto const cur_odm_duration = duration(ride);
          std::println("odm_duration: {}", cur_odm_duration);
          if (cur_odm_duration <= min_odm_duration) {
            min_stop_idx = stop.stop_idx_;
            min_odm_duration = cur_odm_duration;
            shorter_ride = ride;
          }
          break;
        }
      }
    }
    if (shorter_ride) {
      std::println("Found a shorter ODM last leg: [{}, {}, {}]", min_stop_idx,
                   tt.locations_.get(shorter_ride->stop_).name_,
                   min_odm_duration);
      auto& odm_offset = std::get<n::routing::offset>(odm_leg.uses_);
      auto& pt_run_enter_exit =
          std::get<n::routing::journey::run_enter_exit>(pt_leg.uses_);
      pt_run_enter_exit.stop_range_.to_ = pt_run_enter_exit.r_.stop_range_.to_ =
          min_stop_idx + 1U;
      pt_leg.to_ = odm_leg.from_ = odm_offset.target_ = shorter_ride->stop_;
      pt_leg.arr_time_ = odm_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_offset.duration_ = min_odm_duration;
      j.dest_time_ = odm_leg.arr_time_ = shorter_ride->time_at_start_;
    }
  };

  for (auto& j : odm_journeys) {
    if (j.legs_.empty()) {
      fmt::println("shorten_odm: journey without legs");
      continue;
    }
    shorten_first_leg(j);
    shorten_last_leg(j);
  }
}

}  // namespace motis::odm