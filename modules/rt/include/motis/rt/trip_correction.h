#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "utl/get_or_create.h"

#include "motis/core/schedule/schedule.h"

namespace motis::rt {

struct entry;

constexpr auto T_MIN = std::numeric_limits<motis::time>::min();
constexpr auto T_MAX = std::numeric_limits<motis::time>::max();

inline delay_info* get_delay_info(schedule const& sched, ev_key const& k) {
  auto it = sched.graph_to_delay_info_.find(k);
  if (it == end(sched.graph_to_delay_info_)) {
    return nullptr;
  } else {
    return it->second;
  }
}

struct entry : public delay_info {
  entry() = default;
  explicit entry(delay_info const& di) : delay_info{di} {}

  void update_min(time t) { min_ = std::max(min_, t); }
  void update_max(time t) { max_ = std::min(max_, t); }

  void fix() {
    if (get_current_time() > max_) {
      set(timestamp_reason::REPAIR, max_);
    }
    if (get_current_time() < min_) {
      set(timestamp_reason::REPAIR, min_);
    }
    if (get_current_time() == get_original_time()) {
      set(timestamp_reason::REPAIR, 0);
    }
  }

  motis::time min_{T_MIN}, max_{T_MAX};
};

struct trip_corrector {
  explicit trip_corrector(schedule& sched, trip const* trp)
      : sched_(sched),
        trip_ev_keys_(trip_bfs(
            ev_key{trp->edges_->front(), trp->lcon_idx_, event_type::DEP},
            bfs_direction::BOTH)) {}

  std::vector<delay_info const*> fix_times() {
    set_min_max();
    repair();
    return update();
  }

private:
  entry& get_or_create(ev_key const& k) {
    return utl::get_or_create(entries_, k, [&]() {
      auto di_it = sched_.graph_to_delay_info_.find(k);
      if (di_it == end(sched_.graph_to_delay_info_)) {
        delay_info di;
        di.ev_ = k;
        di.orig_ev_ = k;
        di.schedule_time_ = k.get_time();
        return entry(di);
      } else {
        return entry(*di_it->second);
      }
    });
  }

  void apply_is(ev_key const& k, time is) {
    for (auto const& fwd : trip_bfs(k, bfs_direction::FORWARD)) {
      get_or_create(fwd).update_min(is);
    }
    for (auto const& bwd : trip_bfs(k, bfs_direction::BACKWARD)) {
      get_or_create(bwd).update_max(is);
    }
  }

  void set_min_max() {
    for (auto const& k : trip_ev_keys_) {
      auto di = motis::rt::get_delay_info(sched_, k);
      if (di == nullptr || di->get_is_time() == 0) {
        continue;
      }
      apply_is(k, di->get_is_time());
    }
  }

  void repair() {
    for (auto const& k : trip_ev_keys_) {
      entries_[k].fix();
    }
  }

  std::vector<delay_info const*> update() {
    std::vector<delay_info const*> updates;
    for (auto const& k : trip_ev_keys_) {
      auto& e = entries_[k];
      if (e.get_reason() == timestamp_reason::REPAIR &&
          e.get_repair_time() != k.get_time()) {
        auto di = utl::get_or_create(sched_.graph_to_delay_info_, k, [&]() {
          sched_.delay_mem_.emplace_back(mcd::make_unique<delay_info>(k));
          return sched_.delay_mem_.back().get();
        });
        di->set(timestamp_reason::REPAIR, e.get_repair_time());

        auto& event_time = k.ev_type_ == event_type::DEP ? k.lcon()->d_time_
                                                         : k.lcon()->a_time_;
        const_cast<time&>(event_time) = di->get_current_time();  // NOLINT

        updates.push_back(di);
      }
    }
    return updates;
  }

  schedule& sched_;
  std::set<ev_key> trip_ev_keys_;
  std::map<ev_key, entry> entries_;
};

}  // namespace motis::rt
