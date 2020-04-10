#pragma once

#include <string>

#include "motis/hash_map.h"

#include "motis/core/schedule/trip.h"

#include "motis/rt/in_out_allowed.h"

namespace motis::rt {

struct stop_times {
  motis::time arr_, dep_;
  in_out_allowed in_out_;
};

using trip_event_info = mcd::hash_map<std::string /* station id */, stop_times>;

inline trip_event_info get_trip_event_info(schedule const& sched,
                                           trip const* trp) {
  trip_event_info ev;
  for (auto const& trip_e : *trp->edges_) {
    auto const e = trip_e.get_edge();

    auto& dep = ev[sched.stations_.at(e->from_->get_station()->id_)->eva_nr_];
    dep.dep_ = e->get_connection(trp->lcon_idx_)->d_time_;
    dep.in_out_ = get_in_out_allowed(e->from_);

    auto& arr = ev[sched.stations_.at(e->to_->get_station()->id_)->eva_nr_];
    arr.arr_ = e->get_connection(trp->lcon_idx_)->a_time_;
    arr.in_out_ = get_in_out_allowed(e->to_);
  }
  return ev;
}

}  // namespace motis::rt
