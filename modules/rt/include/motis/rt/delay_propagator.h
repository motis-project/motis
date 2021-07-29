#pragma once

#include <queue>
#include <vector>

#include "utl/get_or_create.h"

#include "motis/hash_set.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/event_access.h"
#include "motis/core/access/realtime_access.h"

namespace motis::rt {

struct delay_propagator {
  struct di_cmp {
    inline bool operator()(delay_info const* lhs, delay_info const* rhs) {
      return lhs->get_schedule_time() < rhs->get_schedule_time();
    }
  };

  using pq = std::priority_queue<delay_info*, std::vector<delay_info*>, di_cmp>;

  explicit delay_propagator(schedule& sched) : sched_(sched) {}

  mcd::hash_set<delay_info*> const& events() const { return events_; }

  void add_delay(ev_key const& k, timestamp_reason const reason,
                 time const updated_time) {
    auto di = get_or_create_di(k);
    if (reason != timestamp_reason::SCHEDULE && di->set(reason, updated_time)) {
      expand(di->get_ev_key());
    }
  }

  void recalculate(ev_key const& k) {
    auto const di_it = sched_.graph_to_delay_info_.find(k);
    if (di_it != end(sched_.graph_to_delay_info_)) {
      push(di_it->second);
      expand(k);
    }
  }

  void propagate() {
    while (!pq_.empty()) {
      auto di = pq_.top();
      pq_.pop();

      if (update_propagation(di)) {
        expand(di->get_ev_key());
      }
    }
  }

  void reset() {
    pq_ = pq();
    events_.clear();
  }

private:
  delay_info* get_or_create_di(ev_key const& k) {
    auto di = utl::get_or_create(sched_.graph_to_delay_info_, k, [&]() {
      sched_.delay_mem_.emplace_back(mcd::make_unique<delay_info>(k));
      return sched_.delay_mem_.back().get();
    });
    events_.insert(di);
    return di;
  }

  void push(ev_key const& k) { pq_.push(get_or_create_di(k)); }

  void push(delay_info* di) {
    events_.insert(di);
    pq_.push(di);
  }

  void expand(ev_key const& k) {
    if (k.is_arrival()) {
      for_each_departure(k, [&](ev_key const& dep) { push(dep); });
      auto const& orig_k = get_orig_ev_key(sched_, k);
      auto const dependencies = sched_.trains_wait_for_.find(orig_k);
      if (dependencies != end(sched_.trains_wait_for_)) {
        for (auto const& connector : dependencies->second) {
          push(get_current_ev_key(sched_, connector));
        }
      }
    } else {
      push(k.get_opposite());
    }
  }

  bool update_propagation(delay_info* di) {
    auto k = di->get_ev_key();
    switch (k.ev_type_) {
      case event_type::ARR: {
        // Propagate delay from previous departure.
        auto const dep_di = get_or_create_di(k.get_opposite());
        auto const dep_sched_time = dep_di->get_schedule_time();
        auto const arr_sched_time = di->get_schedule_time();
        auto const duration = arr_sched_time - dep_sched_time;
        auto const propagated = dep_di->get_current_time() + duration;
        return di->set(timestamp_reason::PROPAGATION, propagated);
      }

      case event_type::DEP: {
        auto max = 0;

        // Propagate delay from previous arrivals.
        auto const dep_sched_time = di->get_schedule_time();
        for_each_arrival(k, [&](ev_key const& arr) {
          auto const arr_sched_time = get_schedule_time(sched_, arr);
          auto const sched_standing_time = dep_sched_time - arr_sched_time;
          auto const min_standing = std::min(2, sched_standing_time);
          auto const arr_curr_time = get_or_create_di(arr)->get_current_time();
          max = std::max(max, arr_curr_time + min_standing);
        });

        // Check for dependencies.
        auto const& orig_k = get_orig_ev_key(sched_, k);
        auto const dep_it = sched_.waits_for_trains_.find(orig_k);
        if (dep_it == end(sched_.waits_for_trains_)) {
          return di->set(timestamp_reason::PROPAGATION, max);
        }

        // Propagate delays from dependencies.
        for (auto const& feeder : dep_it->second) {
          auto const current_feeder_k = get_current_ev_key(sched_, feeder);
          if (current_feeder_k.is_canceled()) {
            continue;
          }
          auto const arr_curr_time =
              get_delay_info(sched_, current_feeder_k).get_current_time();
          auto const transfer_time =
              sched_.stations_[k.get_station_idx()]->transfer_time_;
          auto const max_waiting_time =
              sched_.waiting_time_rules_.waiting_time_family(
                  k.lcon()->full_con_->con_info_->family_,
                  current_feeder_k.lcon()->full_con_->con_info_->family_);
          if (arr_curr_time + transfer_time <=
              dep_sched_time + max_waiting_time) {
            max = std::max(max, arr_curr_time + transfer_time);
          }
        }

        return di->set(timestamp_reason::PROPAGATION, max);
      }

      default: return false;
    }
  }

  pq pq_;
  mcd::hash_set<delay_info*> events_;
  schedule& sched_;
};

}  // namespace motis::rt
