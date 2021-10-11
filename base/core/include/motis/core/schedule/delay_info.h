#pragma once

#include <ostream>

#include "motis/core/schedule/event.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timestamp_reason.h"

namespace motis {

struct delay_info {
  inline time get_schedule_time() const { return schedule_time_; }
  inline time get_is_time() const { return is_time_; }
  inline time get_repair_time() const { return repair_time_; }
  inline time get_forecast_time() const { return forecast_time_; }
  inline time get_propagation_time() const { return propagation_time_; }

  inline void update(delay_info const& d) {
    if (schedule_time_ == 0) {
      *this = d;
    } else {
      if (d.forecast_time_.valid()) {
        forecast_time_ = d.forecast_time_;
      }
      if (d.propagation_time_.valid()) {
        propagation_time_ = d.propagation_time_;
      }
      if (is_time_.valid()) {
        is_time_ = d.is_time_;
      }
    }
  }

  inline bool set(timestamp_reason const r, time const t) {
    auto const before = get_current_time();

    switch (r) {
      case timestamp_reason::REPAIR: repair_time_ = t; break;
      case timestamp_reason::SCHEDULE: schedule_time_ = t; break;
      case timestamp_reason::IS: is_time_ = t; break;
      case timestamp_reason::FORECAST: forecast_time_ = t; break;
      case timestamp_reason::PROPAGATION: propagation_time_ = t; break;
    }

    if (r != timestamp_reason::REPAIR) {
      repair_time_ = {};
    }

    return get_current_time() != before;
  }

  inline time get(timestamp_reason const r) const {
    switch (r) {
      case timestamp_reason::REPAIR: return repair_time_;
      case timestamp_reason::IS: return is_time_;
      case timestamp_reason::SCHEDULE: return schedule_time_;
      case timestamp_reason::PROPAGATION: return propagation_time_;
      case timestamp_reason::FORECAST: return forecast_time_;
      default: return schedule_time_;
    }
  }

  inline time get_current_time() const { return get(get_reason()); }

  inline time get_original_time() const { return get(get_original_reason()); }

  inline timestamp_reason get_reason() const {
    if (repair_time_.valid()) {
      return timestamp_reason::REPAIR;
    }
    return get_original_reason();
  }

  inline timestamp_reason get_original_reason() const {
    if (is_time_.valid()) {
      return timestamp_reason::IS;
    } else {
      auto const times = {
          std::make_pair(schedule_time_, timestamp_reason::SCHEDULE),
          std::make_pair(forecast_time_, timestamp_reason::FORECAST),
          std::make_pair(propagation_time_, timestamp_reason::PROPAGATION)};
      return std::max_element(begin(times), end(times))->second;
    }
  }

  inline ev_key const& get_ev_key() const { return ev_; }
  inline ev_key const& get_orig_ev_key() const { return orig_ev_; }

  inline void set_ev_key(ev_key const& k) { ev_ = k; }

  ev_key ev_, orig_ev_{ev_};
  time repair_time_{0U}, is_time_{0U};
  time schedule_time_{ev_.get_time()}, forecast_time_{0U},
      propagation_time_{0U};
};

}  // namespace motis
