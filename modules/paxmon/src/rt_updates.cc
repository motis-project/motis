#include "motis/paxmon/rt_updates.h"

#include <numeric>
#include <set>

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/module/context/motis_parallel_for.h"

#include "motis/paxmon/checks.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/print_stats.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/track_update.h"
#include "motis/paxmon/update_load.h"

using namespace motis::rt;
using namespace motis::logging;
using namespace motis::module;

namespace motis::paxmon {

void check_broken_interchanges(
    universe& uv, std::vector<edge_index> const& updated_interchange_edges,
    int arrival_delay_threshold) {
  static std::set<edge*> broken_interchanges;
  static std::set<passenger_group*> affected_passenger_groups;
  for (auto& icei : updated_interchange_edges) {
    auto* ice = icei.get(uv);
    if (ice->type_ != edge_type::INTERCHANGE) {
      continue;
    }
    auto const from = ice->from(uv);
    auto const to = ice->to(uv);
    auto const ic = static_cast<int>(to->time_) - static_cast<int>(from->time_);
    if (ice->is_canceled(uv) || (from->station_ != 0 && to->station_ != 0 &&
                                 ic < ice->transfer_time())) {
      if (ice->broken_) {
        continue;
      }
      ice->broken_ = true;
      if (broken_interchanges.insert(ice).second) {
        ++uv.system_stats_.total_broken_interchanges_;
      }
      for (auto pg_id : uv.pax_connection_info_.groups_[ice->pci_]) {
        auto* grp = uv.passenger_groups_[pg_id];
        if (affected_passenger_groups.insert(grp).second) {
          uv.system_stats_.total_affected_passengers_ += grp->passengers_;
          grp->ok_ = false;
        }
        uv.rt_update_ctx_.groups_affected_by_last_update_.insert(grp->id_);
      }
    } else if (ice->broken_) {
      // interchange valid again
      ice->broken_ = false;
      for (auto pg_id : uv.pax_connection_info_.groups_[ice->pci_]) {
        auto* grp = uv.passenger_groups_[pg_id];
        uv.rt_update_ctx_.groups_affected_by_last_update_.insert(grp->id_);
      }
    } else if (arrival_delay_threshold >= 0 && to->station_ == 0) {
      // check for delayed arrival at destination
      auto const estimated_arrival = static_cast<int>(from->schedule_time());
      for (auto pg_id : uv.pax_connection_info_.groups_[ice->pci_]) {
        auto* grp = uv.passenger_groups_[pg_id];
        auto const estimated_delay =
            estimated_arrival - static_cast<int>(grp->planned_arrival_time_);
        if (grp->planned_arrival_time_ != INVALID_TIME &&
            estimated_delay >= arrival_delay_threshold) {
          uv.rt_update_ctx_.groups_affected_by_last_update_.insert(grp->id_);
        }
      }
    }
  }
}

void handle_rt_update(universe& uv, capacity_maps const& caps,
                      schedule const& sched, RtUpdates const* update,
                      int arrival_delay_threshold) {
  uv.tick_stats_.rt_updates_ += update->updates()->size();

  std::vector<edge_index> updated_interchange_edges;
  for (auto const& u : *update->updates()) {
    switch (u->content_type()) {
      case Content_RtDelayUpdate: {
        ++uv.system_stats_.delay_updates_;
        ++uv.tick_stats_.rt_delay_updates_;
        auto const du = reinterpret_cast<RtDelayUpdate const*>(u->content());
        update_event_times(sched, uv, du, updated_interchange_edges);
        uv.tick_stats_.rt_delay_event_updates_ += du->events()->size();
        for (auto const& uei : *du->events()) {
          switch (uei->reason()) {
            case TimestampReason_IS:
              ++uv.tick_stats_.rt_delay_is_updates_;
              break;
            case TimestampReason_FORECAST:
              ++uv.tick_stats_.rt_delay_forecast_updates_;
              break;
            case TimestampReason_PROPAGATION:
              ++uv.tick_stats_.rt_delay_propagation_updates_;
              break;
            case TimestampReason_REPAIR:
              ++uv.tick_stats_.rt_delay_repair_updates_;
              break;
            case TimestampReason_SCHEDULE:
              ++uv.tick_stats_.rt_delay_schedule_updates_;
              break;
          }
        }
        break;
      }
      case Content_RtRerouteUpdate: {
        ++uv.system_stats_.reroute_updates_;
        ++uv.tick_stats_.rt_reroute_updates_;
        auto const ru = reinterpret_cast<RtRerouteUpdate const*>(u->content());
        update_trip_route(sched, caps, uv, ru, updated_interchange_edges);
        break;
      }
      case Content_RtTrackUpdate: {
        ++uv.tick_stats_.rt_track_updates_;
        auto const tu = reinterpret_cast<RtTrackUpdate const*>(u->content());
        update_track(sched, uv, tu, updated_interchange_edges);
        break;
      }
      case Content_RtFreeTextUpdate: {
        ++uv.tick_stats_.rt_free_text_updates_;
        break;
      }
      default: break;
    }
  }
  check_broken_interchanges(uv, updated_interchange_edges,
                            arrival_delay_threshold);
}

monitoring_event_type get_monitoring_event_type(
    passenger_group const* pg, reachability_info const& reachability,
    int const arrival_delay_threshold) {
  if (!reachability.ok_) {
    return monitoring_event_type::TRANSFER_BROKEN;
  } else if (arrival_delay_threshold >= 0 &&
             pg->planned_arrival_time_ != INVALID_TIME &&
             pg->estimated_delay() >= arrival_delay_threshold) {
    return monitoring_event_type::MAJOR_DELAY_EXPECTED;
  } else {
    return monitoring_event_type::NO_PROBLEM;
  }
}

std::vector<msg_ptr> update_affected_groups(universe& uv, schedule const& sched,
                                            int arrival_delay_threshold,
                                            int preparation_time) {
  scoped_timer timer{"update affected passenger groups"};
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME,
              "paxmon::update_affected_groups: invalid current system time: "
              "system_time={}, schedule_begin={}",
              sched.system_time_, sched.schedule_begin_);
  auto const search_time = static_cast<time>(current_time + preparation_time);

  uv.tick_stats_.system_time_ = sched.system_time_;

  auto const affected_passenger_count =
      std::accumulate(begin(uv.rt_update_ctx_.groups_affected_by_last_update_),
                      end(uv.rt_update_ctx_.groups_affected_by_last_update_),
                      0ULL, [&](auto const sum, auto const pgi) {
                        return sum + uv.passenger_groups_.at(pgi)->passengers_;
                      });

  uv.tick_stats_.affected_groups_ =
      uv.rt_update_ctx_.groups_affected_by_last_update_.size();
  uv.tick_stats_.affected_passengers_ = affected_passenger_count;

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonEvent>> fbs_events;
  std::vector<msg_ptr> messages;

  std::mutex update_mutex;
  auto total_reachability = 0ULL;
  auto total_localization = 0ULL;
  auto total_update_load = 0ULL;
  auto total_fbs_events = 0ULL;

  auto const print_timing = [&]() {
    if (fbs_events.empty()) {
      return;
    }
    LOG(info) << "update groups timing: reachability "
              << (total_reachability / 1000) << "ms, localization "
              << (total_localization / 1000) << "ms, update_load "
              << (total_update_load / 1000) << "ms, fbs_events "
              << (total_fbs_events / 1000) << "ms, " << fbs_events.size()
              << " events";
  };

  auto const make_monitoring_msg = [&]() {
    if (fbs_events.empty()) {
      return;
    }
    mc.create_and_finish(
        MsgContent_PaxMonUpdate,
        CreatePaxMonUpdate(mc, uv.id_, mc.CreateVector(fbs_events)).Union(),
        "/paxmon/monitoring_update");
    messages.emplace_back(make_msg(mc));
    fbs_events.clear();
    mc.Clear();
  };

  motis_parallel_for(
      uv.rt_update_ctx_.groups_affected_by_last_update_, [&](auto const pgi) {
        auto const& pg = uv.passenger_groups_.at(pgi);
        MOTIS_START_TIMING(reachability);
        auto const reachability =
            get_reachability(uv, pg->compact_planned_journey_);
        pg->ok_ = reachability.ok_;
        if (reachability.ok_) {
          pg->estimated_delay_ = static_cast<std::int16_t>(
              static_cast<int>(
                  reachability.reachable_trips_.back().exit_real_time_) -
              static_cast<int>(pg->planned_arrival_time_));
        }
        MOTIS_STOP_TIMING(reachability);

        MOTIS_START_TIMING(localization);
        auto const localization = localize(sched, reachability, search_time);
        MOTIS_STOP_TIMING(localization);

        auto const event_type = get_monitoring_event_type(
            pg, reachability, arrival_delay_threshold);
        auto const expected_arrival_time =
            event_type == monitoring_event_type::TRANSFER_BROKEN
                ? INVALID_TIME
                : reachability.reachable_trips_.back().exit_real_time_;

        MOTIS_START_TIMING(update_load);
        update_load(pg, reachability, localization, uv);
        MOTIS_STOP_TIMING(update_load);

        MOTIS_START_TIMING(fbs_events);
        std::lock_guard guard{update_mutex};
        fbs_events.emplace_back(to_fbs(
            sched, mc,
            monitoring_event{event_type, *pg, localization,
                             reachability.status_, expected_arrival_time}));
        if (fbs_events.size() >= 10'000) {
          make_monitoring_msg();
        }
        MOTIS_STOP_TIMING(fbs_events);

        total_reachability += MOTIS_TIMING_US(reachability);
        total_localization += MOTIS_TIMING_US(localization);
        total_update_load += MOTIS_TIMING_US(update_load);
        total_fbs_events += MOTIS_TIMING_US(fbs_events);

        if (fbs_events.size() % 10000 == 0) {
          print_timing();
        }

        switch (event_type) {
          case monitoring_event_type::NO_PROBLEM:
            ++uv.tick_stats_.ok_groups_;
            ++uv.system_stats_.groups_ok_count_;
            break;
          case monitoring_event_type::TRANSFER_BROKEN:
            ++uv.tick_stats_.broken_groups_;
            ++uv.system_stats_.groups_broken_count_;
            uv.tick_stats_.broken_passengers_ += pg->passengers_;
            break;
          case monitoring_event_type::MAJOR_DELAY_EXPECTED:
            ++uv.tick_stats_.major_delay_groups_;
            ++uv.system_stats_.groups_major_delay_count_;
            uv.tick_stats_.major_delay_passengers_ += pg->passengers_;
            break;
        }
      });

  print_timing();
  make_monitoring_msg();

  uv.tick_stats_.t_reachability_ = total_reachability / 1000;
  uv.tick_stats_.t_localization_ = total_localization / 1000;
  uv.tick_stats_.t_update_load_ = total_update_load / 1000;
  uv.tick_stats_.t_fbs_events_ = total_fbs_events / 1000;

  return messages;
}

}  // namespace motis::paxmon
