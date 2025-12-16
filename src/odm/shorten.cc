#include "motis/odm/odm.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/odm/prima.h"
#include "motis/transport_mode_ids.h"

using namespace std::chrono_literals;
namespace n = nigiri;
namespace nr = nigiri::routing;

namespace motis::odm {

void shorten(std::vector<nr::journey>& odm_journeys,
             std::vector<nr::offset> const& first_mile_taxi,
             std::vector<service_times_t> const& first_mile_taxi_times,
             std::vector<nr::offset> const& last_mile_taxi,
             std::vector<service_times_t> const& last_mile_taxi_times,
             n::timetable const& tt,
             n::rt_timetable const* rtt,
             api::plan_params const& query) {

  auto const shorten_first_leg = [&](nr::journey& j) {
    auto& odm_leg = begin(j.legs_)[0];
    auto& pt_leg = begin(j.legs_)[1];

    if (!is_odm_leg(odm_leg, kOdmTransportModeId) ||
        !std::holds_alternative<nr::journey::run_enter_exit>(pt_leg.uses_)) {
      return;
    }

    auto& ree = std::get<nr::journey::run_enter_exit>(pt_leg.uses_);
    auto run = n::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.to_ = ree.stop_range_.to_ - 1U;
    auto min_stop_idx = ree.stop_range_.from_;
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<nr::start>{};
    for (auto const stop : run) {
      if (stop.is_cancelled() ||
          !stop.in_allowed(query.pedestrianProfile_ ==
                           api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kDep)) ||
          (query.requireCarTransport_ &&
           !stop.cars_allowed(n::event_type::kDep))) {
        continue;
      }
      for (auto const [offset, times] :
           utl::zip(first_mile_taxi, first_mile_taxi_times)) {
        if (nr::matches(tt, nr::location_match_mode::kExact, offset.target_,
                        stop.get_location_idx()) &&
            utl::any_of(times, [&](auto const& t) {
              return t.contains(stop.time(n::event_type::kDep) -
                                offset.duration_) &&
                     t.contains(stop.time(n::event_type::kDep) - 1min);
            })) {
          if (offset.duration_ < min_odm_duration) {
            min_stop_idx = stop.stop_idx_;
            min_odm_duration = offset.duration_;
            shorter_ride = {.time_at_start_ = stop.time(n::event_type::kDep) -
                                              offset.duration_,
                            .time_at_stop_ = stop.time(n::event_type::kDep),
                            .stop_ = offset.target_};
          }
          break;
        }
      }
    }
    if (shorter_ride) {
      auto& odm_offset = std::get<nr::offset>(odm_leg.uses_);

      auto const old_stop = odm_leg.to_;
      auto const old_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const old_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      j.start_time_ = odm_leg.dep_time_ = shorter_ride->time_at_start_;
      odm_offset.duration_ = min_odm_duration;
      odm_leg.arr_time_ = pt_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_leg.to_ = odm_offset.target_ = pt_leg.from_ = shorter_ride->stop_;
      ree.stop_range_.from_ = min_stop_idx;

      auto const new_stop = odm_leg.to_;
      auto const new_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const new_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      n::log(n::log_lvl::debug, "motis.prima",
             "shorten first leg: [stop: {}, ODM: {}, PT: {}] -> [stop: {}, "
             "ODM: {}, PT: {}] (ODM: -{}, PT: +{})",
             n::loc{tt, old_stop}, old_odm_time, old_pt_time,
             n::loc{tt, new_stop}, new_odm_time, new_pt_time,
             std::chrono::minutes{old_odm_time - new_odm_time},
             new_pt_time - old_pt_time);
    }
  };

  auto const shorten_last_leg = [&](nr::journey& j) {
    auto& odm_leg = rbegin(j.legs_)[0];
    auto& pt_leg = rbegin(j.legs_)[1];

    if (!is_odm_leg(odm_leg, kOdmTransportModeId) ||
        !std::holds_alternative<nr::journey::run_enter_exit>(pt_leg.uses_)) {
      return;
    }

    auto& ree = std::get<nr::journey::run_enter_exit>(pt_leg.uses_);
    auto run = n::rt::frun(tt, rtt, ree.r_);
    run.stop_range_.from_ = ree.stop_range_.from_ + 1U;
    auto min_stop_idx = static_cast<n::stop_idx_t>(ree.stop_range_.to_ - 1U);
    auto min_odm_duration = odm_time(odm_leg);
    auto shorter_ride = std::optional<nr::start>{};
    for (auto const stop : run) {
      if (stop.is_cancelled() ||
          !stop.out_allowed(query.pedestrianProfile_ ==
                            api::PedestrianProfileEnum::WHEELCHAIR) ||
          (query.requireBikeTransport_ &&
           !stop.bikes_allowed(n::event_type::kArr)) ||
          (query.requireCarTransport_ &&
           !stop.cars_allowed(n::event_type::kArr))) {
        continue;
      }
      for (auto const [offset, times] :
           utl::zip(last_mile_taxi, last_mile_taxi_times)) {
        if (nr::matches(tt, nr::location_match_mode::kExact, offset.target_,
                        stop.get_location_idx()) &&
            utl::any_of(times, [&](auto const& t) {
              return t.contains(stop.time(n::event_type::kArr)) &&
                     t.contains(stop.time(n::event_type::kArr) +
                                offset.duration_ - 1min);
            })) {
          if (offset.duration_ < min_odm_duration) {
            min_stop_idx = stop.stop_idx_;
            min_odm_duration = offset.duration_;
            shorter_ride = {.time_at_start_ = stop.time(n::event_type::kArr) +
                                              offset.duration_,
                            .time_at_stop_ = stop.time(n::event_type::kArr),
                            .stop_ = offset.target_};
          }
          break;
        }
      }
    }
    if (shorter_ride) {
      auto& odm_offset = std::get<nr::offset>(odm_leg.uses_);

      auto const old_stop = odm_leg.from_;
      auto const old_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const old_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      ree.stop_range_.to_ = min_stop_idx + 1U;
      pt_leg.to_ = odm_leg.from_ = odm_offset.target_ = shorter_ride->stop_;
      pt_leg.arr_time_ = odm_leg.dep_time_ = shorter_ride->time_at_stop_;
      odm_offset.duration_ = min_odm_duration;
      j.dest_time_ = odm_leg.arr_time_ = shorter_ride->time_at_start_;

      auto const new_stop = odm_leg.from_;
      auto const new_odm_time = std::chrono::minutes{odm_offset.duration_};
      auto const new_pt_time = pt_leg.arr_time_ - pt_leg.dep_time_;

      n::log(n::log_lvl::debug, "motis.prima",
             "shorten last leg: [stop: {}, ODM: {}, PT: {}] -> [stop: {}, "
             "ODM: {}, PT: {}] (ODM: -{}, PT: +{})",
             n::loc{tt, old_stop}, old_odm_time, old_pt_time,
             n::loc{tt, new_stop}, new_odm_time, new_pt_time,
             std::chrono::minutes{old_odm_time - new_odm_time},
             new_pt_time - old_pt_time);
    }
  };

  for (auto& j : odm_journeys) {
    if (j.legs_.empty()) {
      n::log(n::log_lvl::debug, "motis.prima", "shorten: journey without legs");
      continue;
    }
    shorten_first_leg(j);
    shorten_last_leg(j);
  }
}

}  // namespace motis::odm
