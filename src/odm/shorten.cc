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

    auto& ree = std::get<nr::journey::run_enter_exit>(pt_leg.uses_);
    auto run = n::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.to_ = ree.stop_range_.to_ - 1U;
    auto min_stop_idx = ree.stop_range_.from_;
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<n::routing::start>{};
    for (auto const stop : run) {
      if (stop.is_cancelled() ||
          !stop.in_allowed(query.pedestrianProfile_ ==
                           api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kDep))) {
        continue;
      }
      for (auto& ride : from_rides) {
        if (n::routing::matches(tt, n::routing::location_match_mode::kExact,
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
      auto& odm_offset = std::get<n::routing::offset>(odm_leg.uses_);

      auto const old_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const old_stop = tt.locations_.get(odm_leg.to_).name_;
      auto const old_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      j.start_time_ = odm_leg.dep_time_ = shorter_ride->time_at_start_;
      odm_offset.duration_ = min_odm_duration;
      odm_leg.arr_time_ = pt_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_leg.to_ = odm_offset.target_ = pt_leg.from_ = shorter_ride->stop_;
      ree.stop_range_.from_ = min_stop_idx;

      auto const new_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const new_stop = tt.locations_.get(odm_leg.to_).name_;
      auto const new_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      fmt::println(
          "Shortened ODM first leg: [ODM: {}, stop: {}, PT: {}] --> [ODM: {}, "
          "stop: {}, PT: {}] (ODM: -{}, PT: +{})",
          old_odm_time, old_stop, old_pt_time, new_odm_time, new_stop,
          new_pt_time, std::chrono::minutes{old_odm_time - new_odm_time},
          new_pt_time - old_pt_time);
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

    auto& ree = std::get<n::routing::journey::run_enter_exit>(pt_leg.uses_);
    auto run = nigiri::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.from_ = ree.stop_range_.from_ + 1U;
    auto min_stop_idx = static_cast<n::stop_idx_t>(ree.stop_range_.to_ - 1U);
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<n::routing::start>{};
    for (auto const stop : run) {
      if (stop.is_cancelled() ||
          !stop.out_allowed(query.pedestrianProfile_ ==
                            api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kArr))) {
        continue;
      }
      for (auto& ride : to_rides) {
        if (n::routing::matches(tt, n::routing::location_match_mode::kExact,
                                ride.stop_, stop.get_location_idx()) &&
            ride.time_at_stop_ == stop.time(n::event_type::kArr)) {
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
      auto& odm_offset = std::get<n::routing::offset>(odm_leg.uses_);

      auto const old_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const old_stop = tt.locations_.get(odm_leg.from_).name_;
      auto const old_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      ree.stop_range_.to_ = min_stop_idx + 1U;
      pt_leg.to_ = odm_leg.from_ = odm_offset.target_ = shorter_ride->stop_;
      pt_leg.arr_time_ = odm_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_offset.duration_ = min_odm_duration;
      j.dest_time_ = odm_leg.arr_time_ = shorter_ride->time_at_start_;

      auto const new_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const new_stop = tt.locations_.get(odm_leg.from_).name_;
      auto const new_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      fmt::println(
          "Shortened ODM last leg: [ODM: {}, stop: {}, PT: {}] --> [ODM: {}, "
          "stop: {}, PT: {}] (ODM: -{}, PT: +{})",
          old_odm_time, old_stop, old_pt_time, new_odm_time, new_stop,
          new_pt_time, std::chrono::minutes{old_odm_time - new_odm_time},
          new_pt_time - old_pt_time);
    }
  };

  for (auto& j : odm_journeys) {
    if (j.legs_.empty()) {
      fmt::println("shorten: journey without legs");
      continue;
    }
    shorten_first_leg(j);
    shorten_last_leg(j);
  }
}

}  // namespace motis::odm