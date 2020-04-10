#pragma once

#include "motis/core/schedule/event.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

namespace motis::revise {

enum class stop_type_t { TRIP_STOP, WALK_STOP };

struct stop {
  stop(journey::stop old_stop, bool const arr_valid, bool const dep_valid)
      : arr_valid_{arr_valid},
        dep_valid_{dep_valid},
        old_stop_{std::move(old_stop)} {}
  stop(bool const exit, bool const enter, bool const arr_valid,
       bool const dep_valid)
      : exit_{exit},
        enter_{enter},
        arr_valid_{arr_valid},
        dep_valid_{dep_valid} {}

  virtual ~stop() = default;

  stop(stop const&) = delete;
  stop& operator=(stop const&) = delete;
  stop(stop&&) = delete;
  stop& operator=(stop&&) = delete;

  virtual journey::stop get_stop(schedule const& sched) const = 0;
  virtual void propagate_time(schedule const& sched, stop const& pred) = 0;
  virtual void propagate_time_bwd(schedule const& sched, stop& next) = 0;
  virtual stop_type_t get_type() const = 0;
  virtual void set_dep_times(schedule const& sched, ev_key const& k) = 0;
  virtual void set_arr_times(schedule const& sched, ev_key const& k) = 0;
  virtual ev_key get_arr() const = 0;
  virtual ev_key get_dep() const = 0;
  virtual void set_arr(ev_key const& arr) = 0;
  virtual void set_dep(ev_key const& dep) = 0;
  virtual timestamp_reason get_timestamp_reason(schedule const& sched,
                                                event_type type) const = 0;
  virtual void set_timestamp_reason(event_type type,
                                    timestamp_reason reason) = 0;

  void set_exit(bool const exit) { exit_ = exit; }
  void set_enter(bool const enter) { enter_ = enter; }

  time arr_time_{0}, dep_time_{0};
  time arr_sched_time_{0}, dep_sched_time_{0};
  bool exit_{false}, enter_{false};
  bool arr_valid_{false}, dep_valid_{false};
  journey::stop old_stop_;
};

using stop_ptr = std::unique_ptr<stop>;

}  // namespace motis::revise
