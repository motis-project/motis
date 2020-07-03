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
#include "motis/paxmon/loader/csv/csv_journeys.h"
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
  param(journey_files_, "journeys", "csv journeys or routing responses");
  param(capacity_files_, "capacity", "train capacities");
  param(stats_file_, "stats", "statistics file");
  param(start_time_, "start_time", "evaluation start time");
  param(end_time_, "end_time", "evaluation end time");
  param(time_step_, "time_step", "evaluation time step (seconds)");
  param(data_.default_capacity_, "default_capacity", "default capacity");
}

paxmon::~paxmon() = default;

void paxmon::init(motis::module::registry& reg) {
  LOG(info) << "paxmon module loaded";

  stats_writer_ = std::make_unique<stats_writer>(stats_file_);

  add_shared_data(DATA_KEY, &data_);

  reg.subscribe("/init", [&]() {
    load_capacity_files();
    load_journeys();
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

        for (auto t = start_time_; t <= end_time_; t += time_step_) {
          forward(t);
        }

        motis_call(make_no_msg("/paxmon/flush"))->val();

        return {};
      },
      ctx::access_t::WRITE);
}

std::size_t paxmon::load_journeys(std::string const& file) {
  auto const journey_path = fs::path{file};
  if (!fs::exists(journey_path)) {
    LOG(warn) << "journey file not found: " << file;
    return 0;
  }
  auto const& sched = get_schedule();
  std::size_t loaded = 0;
  if (journey_path.extension() == ".txt") {
    scoped_timer journey_timer{"load motis journeys"};
    loaded = loader::journeys::load_journeys(sched, data_, file);
  } else if (journey_path.extension() == ".csv") {
    scoped_timer journey_timer{"load csv journeys"};
    loaded = loader::csv::load_journeys(sched, data_, file);
  } else {
    LOG(logging::error) << "paxmon: unknown journey file type: " << file;
  }
  LOG(loaded != 0 ? info : warn)
      << "loaded " << loaded << " journeys from " << file;
  return loaded;
}

void paxmon::load_journeys() {
  auto const& sched = get_schedule();

  if (journey_files_.empty()) {
    LOG(warn) << "paxmon: no journey files specified";
    return;
  }

  for (auto const& file : journey_files_) {
    load_journeys(file);
  }

  auto build_stats = build_graph_from_journeys(sched, data_);

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
                           build_stats.initial_over_capacity_);
}

void paxmon::load_capacity_files() {
  auto const& sched = get_schedule();
  auto total_entries = 0ULL;
  for (auto const& file : capacity_files_) {
    auto const capacity_path = fs::path{file};
    if (!fs::exists(capacity_path)) {
      LOG(warn) << "capacity file not found: " << file;
      continue;
    }
    auto const entries_loaded = load_capacities(
        sched, file, data_.trip_capacity_map_, data_.category_capacity_map_);
    total_entries += entries_loaded;
    LOG(info) << fmt::format("loaded {:n} capacity entries from {}",
                             entries_loaded, file);
  }
  if (total_entries == 0) {
    LOG(warn) << "no capacity data loaded, using default capacity of "
              << data_.default_capacity_ << " for all trains";
  }
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
