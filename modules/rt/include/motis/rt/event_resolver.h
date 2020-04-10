#pragma once

#include <map>
#include <set>
#include <vector>

#include "boost/optional.hpp"

#include "utl/to_vec.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/edge_access.h"
#include "motis/core/conv/event_type_conv.h"

#include "motis/rt/find_trip_fuzzy.h"
#include "motis/rt/statistics.h"

namespace motis::rt {

struct event_info {
  event_info(uint32_t station_idx, time sched_time, event_type ev_type)
      : station_idx_(station_idx), sched_time_(sched_time), ev_type_(ev_type) {}

  friend bool operator<(event_info const& a, event_info const& b) {
    // For correct event chains: arrivals first.
    auto const a_is_dep = (a.ev_type_ == event_type::DEP);
    auto const b_is_dep = (b.ev_type_ == event_type::DEP);
    return std::tie(a.sched_time_, a.station_idx_, a_is_dep) <
           std::tie(b.sched_time_, b.station_idx_, b_is_dep);
  }

  uint32_t station_idx_;
  time sched_time_;
  event_type ev_type_;
};

inline std::vector<boost::optional<event_info>> resolve_event_info(
    statistics& stats, schedule const& sched,
    std::vector<ris::Event const*> const& events) {
  return utl::to_vec(
      events, [&](ris::Event const* ev) -> boost::optional<event_info> {
        auto const station = find_station(sched, ev->station_id()->str());
        if (station == nullptr) {
          ++stats.ev_station_not_found_;
          return {};
        }

        auto const time = unix_to_motistime(sched, ev->schedule_time());
        if (time == INVALID_TIME) {
          ++stats.ev_invalid_time_;
          return {};
        }

        return {event_info{station->index_, time, from_fbs(ev->type())}};
      });
}

inline std::map<uint32_t /* station_idx */, bool /* is_unique */>
get_station_unique(trip const* trp) {
  std::map<uint32_t, bool> station_unique;
  for (auto const& trp_e : *trp->edges_) {
    if (station_unique.empty()) {
      station_unique[trp_e.get_edge()->from_->get_station()->id_] = true;
    }

    auto const station_idx = trp_e.get_edge()->to_->get_station()->id_;
    auto const it = station_unique.find(station_idx);
    if (it == end(station_unique)) {
      station_unique.emplace(station_idx, true);
    } else {
      it->second = false;
    }
  }
  return station_unique;
}

inline std::vector<boost::optional<ev_key>> resolve_to_ev_keys(
    schedule const& sched, trip const* trp,
    std::vector<boost::optional<event_info>> const& events) {
  auto const station_unique = get_station_unique(trp);
  auto resolved = std::vector<boost::optional<ev_key>>(events.size());

  auto const set_event = [&](edge const* e, event_type const ev_type) {
    auto const station_idx = get_route_node(*e, ev_type)->get_station()->id_;
    for (auto i = 0UL; i < events.size(); ++i) {
      auto const& ev = events[i];
      if (!ev || ev->ev_type_ != ev_type || ev->station_idx_ != station_idx) {
        continue;
      }

      auto const k = ev_key{e, trp->lcon_idx_, ev_type};
      if (station_unique.at(station_idx)) {
        resolved[i] = k;
      } else {
        auto const diff =
            std::abs(static_cast<int>(ev->sched_time_) -
                     static_cast<int>(get_schedule_time(sched, k)));

        if (diff == 0) {
          resolved[i] = k;
        }
      }
    }
  };

  for (auto const& trp_e : *trp->edges_) {
    auto const e = trp_e.get_edge();
    set_event(e, event_type::DEP);
    set_event(e, event_type::ARR);
  }

  return resolved;
}

inline std::pair<trip const*, std::vector<boost::optional<ev_key>>>
resolve_events_and_trip(statistics& stats, schedule const& sched,
                        ris::IdEvent const* id,
                        std::vector<ris::Event const*> const& evs) {
  stats.total_evs_ += evs.size();
  auto trp = find_trip_fuzzy(stats, sched, id);
  if (trp == nullptr) {
    stats.ev_trp_not_found_ += evs.size();
    return {nullptr, {}};
  }
  return {trp, resolve_to_ev_keys(sched, trp,
                                  resolve_event_info(stats, sched, evs))};
}

inline std::vector<boost::optional<ev_key>> resolve_events(
    statistics& stats, schedule const& sched, ris::IdEvent const* id,
    std::vector<ris::Event const*> const& evs) {
  return resolve_events_and_trip(stats, sched, id, evs).second;
}

inline std::vector<boost::optional<ev_key>> resolve_events(
    statistics& stats, schedule const& sched, trip const* trp,
    std::vector<ris::Event const*> const& evs) {
  return resolve_to_ev_keys(sched, trp, resolve_event_info(stats, sched, evs));
}

}  // namespace motis::rt
