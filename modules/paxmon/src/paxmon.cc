#include "motis/paxmon/paxmon.h"

#include <algorithm>
#include <numeric>
#include <set>

#include "boost/filesystem.hpp"

#include "fmt/format.h"

#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

#include "motis/paxmon/build_graph.h"
#include "motis/paxmon/data_key.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/loader/journeys/motis_journeys.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/update_load.h"

namespace fs = boost::filesystem;

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;

namespace motis::paxmon {

paxmon::paxmon() : module("Passenger Monitoring", "paxmon") {
  param(journey_file_, "journeys", "routing responses");
  param(capacity_file_, "capacity", "train capacities");
  param(stats_file_, "stats", "statistics file");
  param(start_time_, "start_time", "evaluation start time");
  param(end_time_, "end_time", "evaluation  end time");
}

paxmon::~paxmon() = default;

void paxmon::init(motis::module::registry& reg) {
  LOG(info) << "paxmon module loaded";

  stats_writer_ = std::make_unique<stats_writer>(stats_file_);

  add_shared_data(DATA_KEY, &data_);

  reg.subscribe("/init", [&]() { load_journeys(); });
  reg.register_op("/paxmon/load_journeys", [&](msg_ptr const&) -> msg_ptr {
    load_journeys();
    return {};
  });
  reg.register_op("/paxmon/flush", [&](msg_ptr const&) -> msg_ptr {
    stats_writer_->flush();
    return {};
  });
  reg.subscribe("/rt/update",
                [&](msg_ptr const& msg) { return rt_update(msg); });
  reg.subscribe("/rt/graph_updated", [&](msg_ptr const&) {
    scoped_timer t{"paxmon: graph_updated"};
    rt_updates_applied();
    return nullptr;
  });

  // --init /paxmon/eval
  // --paxmon.start_time YYYY-MM-DDTHH:mm
  // --paxmon.end_time YYYY-MM-DDTHH:mm
  reg.register_op(
      "/paxmon/eval",
      [&](msg_ptr const&) -> msg_ptr {
        auto const forward = [](std::time_t time) {
          using namespace motis::ris;
          message_creator fbb;
          fbb.create_and_finish(MsgContent_RISForwardTimeRequest,
                                CreateRISForwardTimeRequest(fbb, time).Union(),
                                "/ris/forward");
          LOG(info) << "paxmon: forwarding time to: " << format_unix_time(time);
          motis_call(make_msg(fbb))->val();
        };

        LOG(info) << "paxmon: start time: " << format_unix_time(start_time_)
                  << ", end time: " << format_unix_time(end_time_);

        for (auto t = start_time_; t <= end_time_; t += 60) {
          forward(t);
        }

        motis_call(make_no_msg("/paxmon/flush"))->val();

        return {};
      },
      ctx::access_t::WRITE);

  load_capacity_file();
}

void paxmon::load_journeys() {
  auto const& sched = get_schedule();

  auto const journey_path = fs::path{journey_file_};
  if (!fs::exists(journey_path)) {
    LOG(warn) << "journey file not found: " << journey_file_;
    return;
  }
  if (journey_path.extension() == ".txt") {
    scoped_timer journey_timer{"load motis journeys"};
    LOG(info) << "paxmon: loading motis journeys from file: " << journey_file_;
    loader::journeys::load_journeys(sched, data_, journey_file_);
  } else {
    LOG(logging::error) << "paxmon: unknown journey file type: "
                        << journey_file_;
    return;
  }

  {
    scoped_timer build_graph_timer{"build paxmon graph from journeys"};
    build_graph_from_journeys(sched, data_);
  }

  std::uint64_t edge_count = 0;
  std::uint64_t trip_edge_count = 0;
  std::uint64_t interchange_edge_count = 0;
  std::uint64_t wait_edge_count = 0;
  for (auto const& n : data_.graph_.nodes_) {
    edge_count += n->outgoing_edges(data_.graph_).size();
    for (auto const& e : n->outgoing_edges(data_.graph_)) {
      switch (e->type_) {
        case edge_type::TRIP: ++trip_edge_count; break;
        case edge_type::INTERCHANGE: ++interchange_edge_count; break;
        case edge_type::WAIT: ++wait_edge_count; break;
      }
    }
  }

  std::set<std::uint32_t> stations;
  std::set<trip const*> trips;
  for (auto const& n : data_.graph_.nodes_) {
    stations.insert(n->station_);
    for (auto const& e : n->outgoing_edges(data_.graph_)) {
      if (e->get_trip() != nullptr) {
        trips.insert(e->get_trip());
      }
    }
  }
  auto total_passenger_count = 0ULL;
  for (auto const& pg : data_.graph_.passenger_groups_) {
    total_passenger_count += pg->passengers_;
  }

  LOG(info) << fmt::format("{:n} passenger groups",
                           data_.graph_.passenger_groups_.size());
  LOG(info) << fmt::format("{:n} total passengers", total_passenger_count);
  LOG(info) << fmt::format("{:n} graph nodes", data_.graph_.nodes_.size());
  LOG(info) << fmt::format("{:n} trips", data_.graph_.trip_data_.size());
  LOG(info) << fmt::format(
      "{:n} edges: {:n} trip + {:n} interchange + {:n} wait", edge_count,
      trip_edge_count, interchange_edge_count, wait_edge_count);
  LOG(info) << fmt::format("{:n} stations", stations.size());
  LOG(info) << fmt::format("{:n} trips", trips.size());
  LOG(info) << fmt::format("{:n} edges over capacity initially",
                           initial_over_capacity);
}

void paxmon::load_capacity_file() {
  auto const capacity_path = fs::path{capacity_file_};
  if (!fs::exists(capacity_path)) {
    LOG(warn) << "capacity file not found: " << capacity_file_;
    return;
  }
  data_.capacity_map_ = load_capacities(capacity_file_);
  LOG(info) << fmt::format("loaded capacity data for {:n} trains",
                           data_.capacity_map_.size());
}

void check_broken_interchanges(
    paxmon_data& data, schedule const& /*sched*/,
    std::vector<edge*> const& updated_interchange_edges,
    system_statistics& system_stats) {
  static std::set<edge*> broken_interchanges;
  static std::set<passenger_group*> affected_passenger_groups;
  for (auto& ice : updated_interchange_edges) {
    if (ice->type_ != edge_type::INTERCHANGE) {
      continue;
    }
    auto const from = ice->from(data.graph_);
    auto const to = ice->to(data.graph_);
    auto const ic = static_cast<int>(to->time_) - static_cast<int>(from->time_);
    if (ice->is_canceled(data.graph_) ||
        (from->station_ != 0 && to->station_ != 0 &&
         ic < ice->transfer_time())) {
      if (ice->broken_) {
        continue;
      }
      ice->broken_ = true;
      if (broken_interchanges.insert(ice).second) {
        ++system_stats.total_broken_interchanges_;
      }
      for (auto& psi : ice->pax_connection_info_.section_infos_) {
        if (affected_passenger_groups.insert(psi.group_).second) {
          system_stats.total_affected_passengers_ += psi.group_->passengers_;
          psi.group_->ok_ = false;
        }
        data.groups_affected_by_last_update_.insert(psi.group_);
      }
    } else {
      if (!ice->broken_) {
        continue;
      }
      ice->broken_ = false;
      // interchange valid again
      for (auto& psi : ice->pax_connection_info_.section_infos_) {
        data.groups_affected_by_last_update_.insert(psi.group_);
      }
    }
  }
}

msg_ptr paxmon::rt_update(msg_ptr const& msg) {
  auto const& sched = get_schedule();
  auto update = motis_content(RtUpdates, msg);

  tick_stats_.rt_updates_ += update->updates()->size();

  /*
  LOG(info) << fmt::format("received {:n} rt updates",
                           update->updates()->size());
  */
  std::vector<edge*> updated_interchange_edges;
  for (auto const& u : *update->updates()) {
    switch (u->content_type()) {
      case Content_RtDelayUpdate: {
        ++system_stats_.delay_updates_;
        ++tick_stats_.rt_delay_updates_;
        auto const du = reinterpret_cast<RtDelayUpdate const*>(u->content());
        update_event_times(sched, data_.graph_, du, updated_interchange_edges,
                           system_stats_);
        tick_stats_.rt_delay_event_updates_ += du->events()->size();
        for (auto const& uei : *du->events()) {
          switch (uei->reason()) {
            case TimestampReason_IS: ++tick_stats_.rt_delay_is_updates_; break;
            case TimestampReason_FORECAST:
              ++tick_stats_.rt_delay_forecast_updates_;
              break;
            case TimestampReason_PROPAGATION:
              ++tick_stats_.rt_delay_propagation_updates_;
              break;
            case TimestampReason_REPAIR:
              ++tick_stats_.rt_delay_repair_updates_;
              break;
            case TimestampReason_SCHEDULE:
              ++tick_stats_.rt_delay_schedule_updates_;
              break;
          }
        }
        break;
      }
      case Content_RtRerouteUpdate: {
        ++system_stats_.reroute_updates_;
        ++tick_stats_.rt_reroute_updates_;
        auto const ru = reinterpret_cast<RtRerouteUpdate const*>(u->content());
        update_trip_route(sched, data_, ru, updated_interchange_edges,
                          system_stats_);
        break;
      }
      case Content_RtTrackUpdate: {
        ++tick_stats_.rt_track_updates_;
        break;
      }
      case Content_RtFreeTextUpdate: {
        ++tick_stats_.rt_free_text_updates_;
        break;
      }
      default: break;
    }
  }
  check_broken_interchanges(data_, sched, updated_interchange_edges,
                            system_stats_);
  /*
  LOG(info) << fmt::format(
      "D: {:n} msg, {:n} t.e.f., {:n} dep up, "
      "{:n} arr up -- R: {:n} msg, {:n} t.e.f. -- {:n}/{:n} up ice, "
      "{:n} b.t., {:n} passengers",
      delay_updates, update_event_times_trip_edges_found,
      update_event_times_dep_updated, update_event_times_arr_updated,
      reroute_updates, update_trip_route_trip_edges_found,
      updated_interchange_edges.size(), total_updated_interchange_edges,
      total_broken_interchanges, total_affected_passengers);
  */
  return {};
}

void paxmon::rt_updates_applied() {
  auto const& sched = get_schedule();
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");
  auto const preparation_time = 15 /*min*/;
  auto const search_time = current_time + preparation_time;

  tick_stats_.system_time_ = sched.system_time_;

  auto const affected_passenger_count = std::accumulate(
      begin(data_.groups_affected_by_last_update_),
      end(data_.groups_affected_by_last_update_), 0ULL,
      [](auto const sum, auto const& pg) { return sum + pg->passengers_; });

  tick_stats_.affected_groups_ = data_.groups_affected_by_last_update_.size();
  tick_stats_.affected_passengers_ = affected_passenger_count;

  auto ok_groups = 0ULL;
  auto broken_groups = 0ULL;
  auto broken_passengers = 0ULL;
  {
    scoped_timer timer{"update affected passenger groups"};
    message_creator mc;
    std::vector<flatbuffers::Offset<MonitoringEvent>> fbs_events;

    for (auto const pg : data_.groups_affected_by_last_update_) {
      auto const reachability =
          get_reachability(data_, sched, pg->compact_planned_journey_);
      pg->ok_ = reachability.ok_;

      auto const localization = localize(sched, reachability, search_time);
      update_load(pg, reachability, localization, data_.graph_);

      auto const event_type = reachability.ok_
                                  ? monitoring_event_type::NO_PROBLEM
                                  : monitoring_event_type::TRANSFER_BROKEN;
      fbs_events.emplace_back(
          to_fbs(sched, mc,
                 monitoring_event{event_type, *pg, localization,
                                  reachability.status_}));

      if (reachability.ok_) {
        ++ok_groups;
        ++system_stats_.groups_ok_count_;
        continue;
      }
      ++broken_groups;
      ++system_stats_.groups_broken_count_;
      broken_passengers += pg->passengers_;
    }

    mc.create_and_finish(
        MsgContent_MonitoringUpdate,
        CreateMonitoringUpdate(mc, mc.CreateVector(fbs_events)).Union(),
        "/paxmon/monitoring_update");
    ctx::await_all(motis_publish(make_msg(mc)));
  }

  tick_stats_.ok_groups_ = ok_groups;
  tick_stats_.broken_groups_ = broken_groups;
  tick_stats_.broken_passengers_ = broken_passengers;
  tick_stats_.total_ok_groups_ = system_stats_.groups_ok_count_;
  tick_stats_.total_broken_groups_ = system_stats_.groups_broken_count_;

  LOG(info) << "affected by last rt update: "
            << data_.groups_affected_by_last_update_.size()
            << " passenger groups, "
            << " passengers";

  data_.groups_affected_by_last_update_.clear();
  LOG(info) << "passenger groups: " << ok_groups << " ok, " << broken_groups
            << " broken - passengers affected by broken groups: "
            << broken_passengers;
  LOG(info) << "groups: " << system_stats_.groups_ok_count_ << " ok + "
            << system_stats_.groups_broken_count_ << " broken";

  for (auto const& pg : data_.graph_.passenger_groups_) {
    if (pg->ok_) {
      ++tick_stats_.tracked_ok_groups_;
    } else {
      ++tick_stats_.tracked_broken_groups_;
    }
  }

  stats_writer_->write_tick(tick_stats_);
  stats_writer_->flush();
  tick_stats_ = {};
}

}  // namespace motis::paxmon
