#pragma once

#include "motis/core/schedule/event.h"

namespace motis::rt {

inline bool fits_edge(ev_key const& k, motis::time const t) {
  auto prev = k.lcon_idx_ == 0
                  ? ev_key{}  //
                  : ev_key{k.route_edge_, k.lcon_idx_ - 1, k.ev_type_};
  auto succ = k.lcon_idx_ == k.route_edge_->m_.route_edge_.conns_.size() - 1
                  ? ev_key{}
                  : ev_key{k.route_edge_, k.lcon_idx_ + 1, k.ev_type_};
  return (!prev.is_not_null() || t > prev.get_time()) &&
         (!succ.is_not_null() || t < succ.get_time());
}

inline bool fits_edge(schedule const& sched, ev_key const& k,
                      uint16_t const new_track) {
  if (k.route_edge_->m_.route_edge_.conns_.size() == 1) {
    return true;
  }
  auto const station = sched.stations_[k.get_station_idx()].get();
  return station->get_platform(k.is_departure()
                                   ? k.lcon()->full_con_->d_track_
                                   : k.lcon()->full_con_->a_track_) ==
         station->get_platform(new_track);
}

inline bool fits_trip(schedule const& sched, ev_key const& k,
                      motis::time const t) {
  switch (k.ev_type_) {
    case event_type::ARR: {
      auto const dep_time =
          motis::get_delay_info(sched, k.get_opposite()).get_current_time();
      if (dep_time > t) {
        return false;
      }

      bool valid = true;
      for_each_departure(k, [&](ev_key const& dep) {
        valid =
            valid && motis::get_delay_info(sched, dep).get_current_time() >= t;
      });
      return valid;
    }

    case event_type::DEP: {
      auto const arr_time =
          motis::get_delay_info(sched, k.get_opposite()).get_current_time();
      if (t > arr_time) {
        return false;
      }

      bool valid = true;
      for_each_arrival(k, [&](ev_key const& arr) {
        valid =
            valid && motis::get_delay_info(sched, arr).get_current_time() <= t;
      });
      return valid;
    }

    default: return true;
  }
}

}  // namespace motis::rt
