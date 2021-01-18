#include "motis/paxmon/paxmon.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>

#include "boost/filesystem.hpp"

#include "fmt/format.h"

#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/event_collector.h"
#include "motis/module/message.h"

#include "motis/paxmon/broken_interchanges_report.h"
#include "motis/paxmon/build_graph.h"
#include "motis/paxmon/checks.h"
#include "motis/paxmon/data_key.h"
#include "motis/paxmon/generate_capacities.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/loader/csv/csv_journeys.h"
#include "motis/paxmon/loader/journeys/motis_journeys.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/output/journey_converter.h"
#include "motis/paxmon/output/mcfp_scenario.h"
#include "motis/paxmon/over_capacity_report.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/service_info.h"
#include "motis/paxmon/update_load.h"

namespace fs = boost::filesystem;

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;

namespace motis::paxmon {

paxmon::paxmon() : module("Passenger Monitoring", "paxmon") {
  param(generated_capacity_file_, "generated_capacity_file",
        "output for generated capacities");
  param(stats_file_, "stats", "statistics file");
  param(capacity_match_log_file_, "capacity_match_log",
        "capacity match log file");
  param(journey_match_log_file_, "journey_match_log", "journey match log file");
  param(initial_over_capacity_report_file_, "over_capacity_report",
        "initial over capacity report file");
  param(initial_broken_report_file_, "broken_report",
        "initial broken interchanges report file");
  param(reroute_unmatched_, "reroute_unmatched", "reroute unmatched journeys");
  param(initial_reroute_query_file_, "reroute_file",
        "output file for initial rerouted journeys");
  param(initial_reroute_router_, "reroute_router",
        "router for initial reroute queries");
  param(start_time_, "start_time", "evaluation start time");
  param(end_time_, "end_time", "evaluation end time");
  param(time_step_, "time_step", "evaluation time step (seconds)");
  param(match_tolerance_, "match_tolerance",
        "journey match time tolerance (minutes)");
  param(arrival_delay_threshold_, "arrival_delay_threshold",
        "threshold for arrival delay at the destination (minutes, -1 to "
        "disable)");
  param(check_graph_times_, "check_graph_times",
        "check graph timestamps after each update");
  param(check_graph_integrity_, "check_graph_integrity",
        "check graph integrity after each update");
  param(mcfp_scenario_dir_, "mcfp_scenario_dir",
        "output directory for mcfp scenarios");
  param(mcfp_scenario_min_broken_groups_, "mcfp_scenario_min_broken_groups",
        "required number of broken groups in an update for mcfp scenarios");
  param(mcfp_scenario_include_trip_info_, "mcfp_scenario_include_trip_info",
        "include trip info (category + train_nr) in mcfp scenarios");
}

paxmon::~paxmon() = default;

void paxmon::import(motis::module::registry& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "paxmon", reg,
      [this](std::map<std::string, msg_ptr> const& dependencies) {
        using namespace motis::import;
        auto const msg = dependencies.at("PAXMON_DATA");

        for (auto const* ip : *motis_content(FileEvent, msg)->paths()) {
          auto const tag = ip->tag()->str();
          auto const path = ip->path()->str();
          if (tag == "capacity") {
            if (fs::is_regular_file(path)) {
              capacity_files_.emplace_back(path);
            } else {
              LOG(warn) << "capacity file not found: " << path;
              import_successful_ = false;
            }
          } else if (tag == "journeys") {
            if (fs::is_regular_file(path)) {
              journey_files_.emplace_back(path);
            } else {
              LOG(warn) << "journey file not found: " << path;
              import_successful_ = false;
            }
          }
        }

        load_capacity_files();
        load_journeys();
      })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("PAXMON_DATA", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_FileEvent;
      });
}

void paxmon::init(motis::module::registry& reg) {
  stats_writer_ = std::make_unique<stats_writer>(stats_file_);

  add_shared_data(DATA_KEY, &data_);

  reg.subscribe("/init", [&]() {
    if (data_.trip_capacity_map_.empty() &&
        data_.category_capacity_map_.empty()) {
      LOG(warn) << "no capacity information available";
    }
    LOG(info) << "tracking " << data_.graph_.passenger_groups_.size()
              << " passenger groups";
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

  auto const forward = [](std::time_t time) {
    using namespace motis::ris;
    message_creator fbb;
    fbb.create_and_finish(MsgContent_RISForwardTimeRequest,
                          CreateRISForwardTimeRequest(fbb, time).Union(),
                          "/ris/forward");
    LOG(info) << "paxmon: forwarding time to: " << format_unix_time(time)
              << " =========================================";
    return motis_call(make_msg(fbb))->val();
  };

  // --init /paxmon/eval
  // --paxmon.start_time YYYY-MM-DDTHH:mm
  // --paxmon.end_time YYYY-MM-DDTHH:mm
  reg.register_op(
      "/paxmon/eval",
      [&](msg_ptr const&) -> msg_ptr {
        LOG(info) << "paxmon: start time: " << format_unix_time(start_time_)
                  << ", end time: " << format_unix_time(end_time_);

        for (auto t = start_time_; t <= end_time_; t += time_step_) {
          forward(t);
        }

        motis_call(make_no_msg("/paxmon/flush"))->val();

        LOG(info) << "paxmon: eval done";

        return {};
      },
      ctx::access_t::WRITE);

  // --init /paxmon/generate_capacities
  // --paxmon.generated_capacity_file file.csv
  reg.register_op(
      "/paxmon/generate_capacities", [&](msg_ptr const&) -> msg_ptr {
        if (generated_capacity_file_.empty()) {
          LOG(logging::error)
              << "generate_capacities: no output file specified";
          return {};
        }
        generate_capacities(get_schedule(), data_, generated_capacity_file_);
        return {};
      });

  reg.register_op(
      "/paxmon/init_forward",
      [&](msg_ptr const&) -> msg_ptr { return forward(start_time_); },
      ctx::access_t::WRITE);

  reg.register_op("/paxmon/add_groups", [&](msg_ptr const& msg) -> msg_ptr {
    return add_groups(msg);
  });

  reg.register_op("/paxmon/remove_groups", [&](msg_ptr const& msg) -> msg_ptr {
    return remove_groups(msg);
  });

  reg.register_op("/paxmon/trip_load_info", [&](msg_ptr const& msg) -> msg_ptr {
    return get_trip_load_info(msg);
  });

  reg.register_op("/paxmon/find_trips", [&](msg_ptr const& msg) -> msg_ptr {
    return find_trips(msg);
  });

  reg.register_op("/paxmon/status", [&](msg_ptr const& msg) -> msg_ptr {
    return get_status(msg);
  });

  if (!mcfp_scenario_dir_.empty()) {
    if (fs::exists(mcfp_scenario_dir_)) {
      write_mcfp_scenarios_ = fs::is_directory(mcfp_scenario_dir_);
    } else {
      write_mcfp_scenarios_ = fs::create_directories(mcfp_scenario_dir_);
    }
  }
}

void print_graph_stats(graph_statistics const& graph_stats) {
  LOG(info) << fmt::format("{:L} passenger groups, {:L} passengers",
                           graph_stats.passenger_groups_,
                           graph_stats.passengers_);
  LOG(info) << fmt::format("{:L} graph nodes ({:L} canceled)",
                           graph_stats.nodes_, graph_stats.canceled_nodes_);
  LOG(info) << fmt::format(
      "{:L} graph edges ({:L} canceled): {:L} trip + {:L} interchange + {:L} "
      "wait + {:L} through",
      graph_stats.edges_, graph_stats.canceled_edges_, graph_stats.trip_edges_,
      graph_stats.interchange_edges_, graph_stats.wait_edges_,
      graph_stats.through_edges_);
  LOG(info) << fmt::format("{:L} stations", graph_stats.stations_);
  LOG(info) << fmt::format("{:L} trips", graph_stats.trips_);
  LOG(info) << fmt::format("over capacity: {:L} trips, {:L} edges",
                           graph_stats.trips_over_capacity_,
                           graph_stats.edges_over_capacity_);
  LOG(info) << fmt::format("broken: {:L} interchange edges, {:L} groups",
                           graph_stats.broken_edges_,
                           graph_stats.broken_passenger_groups_);
}

void print_allocator_stats(graph const& g) {
  LOG(info) << fmt::format(
      "passenger group allocator: {:L} groups, {:.2f} MiB currently allocated, "
      "{:L} free list entries, {:L} total allocations, {:L} total "
      "deallocations",
      g.passenger_group_allocator_.elements_allocated(),
      static_cast<double>(g.passenger_group_allocator_.bytes_allocated()) /
          (1024.0 * 1024.0),
      g.passenger_group_allocator_.free_list_size(),
      g.passenger_group_allocator_.allocation_count(),
      g.passenger_group_allocator_.release_count());
}

loader::loader_result paxmon::load_journeys(std::string const& file) {
  auto const journey_path = fs::path{file};
  if (!fs::exists(journey_path)) {
    LOG(warn) << "journey file not found: " << file;
    return {};
  }
  auto const& sched = get_schedule();
  auto result = loader::loader_result{};
  if (journey_path.extension() == ".txt") {
    scoped_timer journey_timer{"load motis journeys"};
    result = loader::journeys::load_journeys(sched, data_, file);
  } else if (journey_path.extension() == ".csv") {
    scoped_timer journey_timer{"load csv journeys"};
    result = loader::csv::load_journeys(
        sched, data_, file, journey_match_log_file_, match_tolerance_);
  } else {
    LOG(logging::error) << "paxmon: unknown journey file type: " << file;
  }
  LOG(result.loaded_journeys_ != 0 ? info : warn)
      << "loaded " << result.loaded_journeys_ << " journeys from " << file;
  return result;
}

msg_ptr initial_reroute_query(schedule const& sched,
                              loader::unmatched_journey const& uj,
                              std::string const& router) {
  message_creator fbb;
  auto const planned_departure =
      motis_to_unixtime(sched.schedule_begin_, uj.departure_time_);
  auto const interval = Interval{planned_departure - 2 * 60 * 60,
                                 planned_departure + 2 * 60 * 60};
  auto const& start_station = sched.stations_.at(uj.start_station_idx_);
  auto const& destination_station =
      sched.stations_.at(uj.destination_station_idx_);
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString(start_station->eva_nr_),
                                 fbb.CreateString(start_station->name_)),
              &interval)
              .Union(),
          CreateInputStation(fbb,
                             fbb.CreateString(destination_station->eva_nr_),
                             fbb.CreateString(destination_station->name_)),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<flatbuffers::Offset<Via>>{}),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<AdditionalEdgeWrapper>>{}))
          .Union(),
      router);
  return make_msg(fbb);
}

void paxmon::load_journeys() {
  auto const& sched = get_schedule();
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Load Journeys")
      .out_bounds(10.F, 60.F)
      .in_high(journey_files_.size());

  if (journey_files_.empty()) {
    LOG(warn) << "paxmon: no journey files specified";
    return;
  }

  {
    std::unique_ptr<output::journey_converter> converter;
    if (reroute_unmatched_ && !initial_reroute_query_file_.empty()) {
      converter = std::make_unique<output::journey_converter>(
          initial_reroute_query_file_);
    }
    for (auto const& file : journey_files_) {
      auto const result = load_journeys(file);
      if (reroute_unmatched_) {
        scoped_timer timer{"reroute unmatched journeys"};
        LOG(info) << "routing " << result.unmatched_journeys_.size()
                  << " unmatched journeys using " << initial_reroute_router_
                  << "...";
        auto const futures =
            utl::to_vec(result.unmatched_journeys_, [&](auto const& uj) {
              return motis_call(
                  initial_reroute_query(sched, uj, initial_reroute_router_));
            });
        ctx::await_all(futures);
        LOG(info) << "adding replacement journeys...";
        for (auto const& [uj, fut] :
             utl::zip(result.unmatched_journeys_, futures)) {
          auto const rr_msg = fut->val();
          auto const rr = motis_content(RoutingResponse, rr_msg);
          auto const journeys = message_to_journeys(rr);
          if (journeys.empty()) {
            continue;
          }
          // TODO(pablo): select journey(s)
          if (converter) {
            converter->write_journey(journeys.front(), uj.source_.primary_ref_,
                                     uj.source_.secondary_ref_, uj.passengers_);
          }
          loader::journeys::load_journey(sched, data_, journeys.front(),
                                         uj.source_, uj.passengers_,
                                         group_source_flags::MATCH_REROUTED);
        }
      }
      progress_tracker->increment();
    }
  }

  progress_tracker->status("Build Graph").out_bounds(60.F, 100.F);
  build_graph_from_journeys(sched, data_);

  auto const graph_stats = calc_graph_statistics(sched, data_);
  print_graph_stats(graph_stats);
  print_allocator_stats(data_.graph_);
  if (graph_stats.trips_over_capacity_ > 0 &&
      !initial_over_capacity_report_file_.empty()) {
    write_over_capacity_report(data_, sched,
                               initial_over_capacity_report_file_);
  }
  if (!initial_broken_report_file_.empty()) {
    write_broken_interchanges_report(data_, initial_broken_report_file_);
  }

  if (check_graph_times_) {
    utl::verify(check_graph_times(data_.graph_, sched),
                "load_journeys: check_graph_times");
  }
  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, sched),
                "load_journeys: check_graph_integrity");
  }
}

void paxmon::load_capacity_files() {
  auto const& sched = get_schedule();
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Load Capacity Data")
      .out_bounds(0.F, 10.F)
      .in_high(capacity_files_.size());
  auto total_entries = 0ULL;
  for (auto const& file : capacity_files_) {
    auto const capacity_path = fs::path{file};
    if (!fs::exists(capacity_path)) {
      LOG(warn) << "capacity file not found: " << file;
      import_successful_ = false;
      continue;
    }
    auto const entries_loaded =
        load_capacities(sched, file, data_.trip_capacity_map_,
                        data_.category_capacity_map_, capacity_match_log_file_);
    total_entries += entries_loaded;
    LOG(info) << fmt::format("loaded {:L} capacity entries from {}",
                             entries_loaded, file);
    progress_tracker->increment();
  }
  if (total_entries == 0) {
    LOG(warn)
        << "no capacity data loaded, all trips will have unknown capacity";
  }
}

void check_broken_interchanges(
    paxmon_data& data, schedule const& /*sched*/,
    std::vector<edge*> const& updated_interchange_edges,
    system_statistics& system_stats, int arrival_delay_threshold) {
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
      for (auto grp : ice->pax_connection_info_.groups_) {
        if (affected_passenger_groups.insert(grp).second) {
          system_stats.total_affected_passengers_ += grp->passengers_;
          grp->ok_ = false;
        }
        data.groups_affected_by_last_update_.insert(grp);
      }
    } else if (ice->broken_) {
      // interchange valid again
      ice->broken_ = false;
      for (auto grp : ice->pax_connection_info_.groups_) {
        data.groups_affected_by_last_update_.insert(grp);
      }
    } else if (arrival_delay_threshold < 0 && to->station_ == 0) {
      // check for delayed arrival at destination
      auto const estimated_arrival = static_cast<int>(from->schedule_time());
      for (auto grp : ice->pax_connection_info_.groups_) {
        auto const estimated_delay =
            estimated_arrival - static_cast<int>(grp->planned_arrival_time_);
        if (grp->planned_arrival_time_ != INVALID_TIME &&
            estimated_delay >= arrival_delay_threshold) {
          data.groups_affected_by_last_update_.insert(grp);
        }
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
                            system_stats_, arrival_delay_threshold_);
  return {};
}

monitoring_event_type get_monitoring_event_type(
    passenger_group const* pg, reachability_info const& reachability,
    int const arrival_delay_threshold) {
  if (!reachability.ok_) {
    return monitoring_event_type::TRANSFER_BROKEN;
  } else if (arrival_delay_threshold >= 0 &&
             !reachability.reachable_trips_.empty() &&
             pg->planned_arrival_time_ != INVALID_TIME &&
             (static_cast<int>(
                  reachability.reachable_trips_.back().exit_real_time_) -
                  static_cast<int>(pg->planned_arrival_time_) >=
              arrival_delay_threshold)) {
    return monitoring_event_type::MAJOR_DELAY_EXPECTED;
  } else {
    return monitoring_event_type::NO_PROBLEM;
  }
}

void paxmon::rt_updates_applied() {
  MOTIS_START_TIMING(total);
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

  data_.trips_affected_by_last_update_.clear();

  if (check_graph_times_) {
    utl::verify(check_graph_times(data_.graph_, sched),
                "rt_updates_applied: check_graph_times");
  }
  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, sched),
                "rt_updates_applied: check_graph_integrity (start)");
  }

  auto ok_groups = 0ULL;
  auto broken_groups = 0ULL;
  auto broken_passengers = 0ULL;
  {
    manual_timer timer{"update affected passenger groups"};
    message_creator mc;
    std::vector<flatbuffers::Offset<PaxMonEvent>> fbs_events;
    std::vector<msg_ptr> messages;

    LOG(info) << "groups affected by last update: "
              << data_.groups_affected_by_last_update_.size()
              << ", total groups: " << data_.graph_.passenger_groups_.size();
    print_allocator_stats(data_.graph_);

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
          CreatePaxMonUpdate(mc, mc.CreateVector(fbs_events)).Union(),
          "/paxmon/monitoring_update");
      messages.emplace_back(make_msg(mc));
      fbs_events.clear();
      mc.Clear();
    };

    motis_parallel_for(
        data_.groups_affected_by_last_update_, [&](auto const& pg) {
          MOTIS_START_TIMING(reachability);
          auto const reachability =
              get_reachability(data_, pg->compact_planned_journey_);
          pg->ok_ = reachability.ok_;
          MOTIS_STOP_TIMING(reachability);

          MOTIS_START_TIMING(localization);
          auto const localization = localize(sched, reachability, search_time);
          MOTIS_STOP_TIMING(localization);
          auto const event_type = get_monitoring_event_type(
              pg, reachability, arrival_delay_threshold_);

          MOTIS_START_TIMING(update_load);
          update_load(pg, reachability, localization, data_.graph_);
          MOTIS_STOP_TIMING(update_load);

          MOTIS_START_TIMING(fbs_events);
          std::lock_guard guard{update_mutex};
          fbs_events.emplace_back(
              to_fbs(sched, mc,
                     monitoring_event{event_type, *pg, localization,
                                      reachability.status_}));
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

          if (reachability.ok_) {
            ++ok_groups;
            ++system_stats_.groups_ok_count_;
            return;
          }
          ++broken_groups;
          ++system_stats_.groups_broken_count_;
          broken_passengers += pg->passengers_;
        });

    print_timing();

    make_monitoring_msg();
    timer.stop_and_print();

    if (check_graph_integrity_) {
      utl::verify(
          check_graph_integrity(data_.graph_, sched),
          "rt_updates_applied: check_graph_integrity (after load update)");
    }

    if (write_mcfp_scenarios_ &&
        broken_groups >= mcfp_scenario_min_broken_groups_) {
      auto const dir =
          fs::path{mcfp_scenario_dir_} /
          fs::path{fmt::format(
              "{}_{}", format_unix_time(sched.system_time_, "%Y-%m-%d_%H-%M"),
              broken_groups)};
      LOG(info) << "writing MCFP scenario with " << broken_groups
                << " broken groups to " << dir.string();
      fs::create_directories(dir);
      output::write_scenario(dir, sched, data_, messages,
                             mcfp_scenario_include_trip_info_);
    }

    MOTIS_START_TIMING(publish);
    for (auto& msg : messages) {
      ctx::await_all(motis_publish(msg));
      msg.reset();
    }
    MOTIS_STOP_TIMING(publish);

    tick_stats_.t_reachability_ = total_reachability / 1000;
    tick_stats_.t_localization_ = total_localization / 1000;
    tick_stats_.t_update_load_ = total_update_load / 1000;
    tick_stats_.t_fbs_events_ = total_fbs_events / 1000;
    tick_stats_.t_publish_ =
        static_cast<std::uint64_t>(MOTIS_TIMING_MS(publish));
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
    if (pg == nullptr) {
      continue;
    }
    if (pg->ok_) {
      ++tick_stats_.tracked_ok_groups_;
    } else {
      ++tick_stats_.tracked_broken_groups_;
    }
  }

  MOTIS_STOP_TIMING(total);
  tick_stats_.t_rt_updates_applied_total_ =
      static_cast<std::uint64_t>(MOTIS_TIMING_MS(total));

  stats_writer_->write_tick(tick_stats_);
  stats_writer_->flush();
  last_tick_stats_ = tick_stats_;
  tick_stats_ = {};

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, sched),
                "rt_updates_applied: check_graph_integrity (end)");
  }
}

msg_ptr paxmon::add_groups(msg_ptr const& msg) {
  auto const& sched = get_schedule();
  auto const req = motis_content(PaxMonAddGroupsRequest, msg);

  auto const added_groups =
      utl::to_vec(*req->groups(), [&](PaxMonGroup const* pg_fbs) {
        utl::verify(pg_fbs->planned_journey()->legs()->size() != 0,
                    "trying to add empty passenger group");
        auto const id =
            static_cast<std::uint64_t>(data_.graph_.passenger_groups_.size());
        auto pg = data_.graph_.passenger_groups_.emplace_back(
            data_.graph_.passenger_group_allocator_.create(
                from_fbs(sched, pg_fbs)));
        pg->id_ = id;
        add_passenger_group_to_graph(sched, data_, *pg);
        for (auto const& leg : pg->compact_planned_journey_.legs_) {
          data_.trips_affected_by_last_update_.insert(leg.trip_);
        }
        return pg;
      });

  print_allocator_stats(data_.graph_);

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonAddGroupsResponse,
      CreatePaxMonAddGroupsResponse(
          mc, mc.CreateVector(utl::to_vec(
                  added_groups, [](auto const pg) { return pg->id_; })))
          .Union());
  return make_msg(mc);
}

msg_ptr paxmon::remove_groups(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonRemoveGroupsRequest, msg);

  for (auto const id : *req->ids()) {
    auto& pg = data_.graph_.passenger_groups_.at(id);
    if (pg == nullptr) {
      continue;
    }
    for (auto const& leg : pg->compact_planned_journey_.legs_) {
      data_.trips_affected_by_last_update_.insert(leg.trip_);
    }
    remove_passenger_group_from_graph(pg);
    data_.graph_.passenger_group_allocator_.release(pg);
    pg = nullptr;
  }

  print_allocator_stats(data_.graph_);

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, get_schedule()),
                "remove_groups (end)");
  }

  return {};
}

msg_ptr paxmon::get_trip_load_info(msg_ptr const& msg) {
  auto const req = motis_content(TripId, msg);
  auto const& sched = get_schedule();
  auto const trp = from_fbs(sched, req);

  auto const tli = calc_trip_load_info(data_, trp);
  message_creator mc;
  mc.create_and_finish(MsgContent_PaxMonTripLoadInfo,
                       to_fbs(mc, sched, data_.graph_, tli).Union());
  return make_msg(mc);
}

msg_ptr paxmon::find_trips(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFindTripsRequest, msg);
  auto const& sched = get_schedule();

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonTripInfo>> trips;
  auto const search_entry = std::make_pair(
      primary_trip_id{0U, req->train_nr(), 0U}, static_cast<trip*>(nullptr));
  for (auto it = std::lower_bound(begin(sched.trips_), end(sched.trips_),
                                  search_entry);
       it != end(sched.trips_) && it->first.train_nr_ == req->train_nr();
       ++it) {
    auto const trp = static_cast<trip const*>(it->second);
    if (trp->edges_->empty()) {
      continue;
    }
    auto const has_paxmon_data =
        data_.graph_.trip_data_.find(trp) != end(data_.graph_.trip_data_);
    if (req->only_trips_with_paxmon_data() && !has_paxmon_data) {
      continue;
    }
    auto const service_infos = get_service_infos(sched, trp);
    if (req->filter_class()) {
      if (std::any_of(begin(service_infos), end(service_infos),
                      [&](auto const& p) {
                        return static_cast<service_class_t>(p.first.clasz_) >
                               static_cast<service_class_t>(req->max_class());
                      })) {
        continue;
      }
    }
    trips.emplace_back(CreatePaxMonTripInfo(
        mc, to_fbs_trip_service_info(mc, sched, trp, service_infos),
        has_paxmon_data));
  }

  mc.create_and_finish(
      MsgContent_PaxMonFindTripsResponse,
      CreatePaxMonFindTripsResponse(mc, mc.CreateVector(trips)).Union());
  return make_msg(mc);
}

msg_ptr paxmon::get_status(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonStatusRequest, msg);
  auto const& sched = get_schedule();

  message_creator mc;
  std::vector<flatbuffers::Offset<TripServiceInfo>>
      trips_affected_by_last_update;
  std::vector<flatbuffers::Offset<TripServiceInfo>>
      trips_with_critical_sections;

  if (req->include_trips_affected_by_last_update()) {
    trips_affected_by_last_update.reserve(
        data_.trips_affected_by_last_update_.size());
    for (auto const trp : data_.trips_affected_by_last_update_) {
      trips_affected_by_last_update.emplace_back(
          to_fbs_trip_service_info(mc, sched, trp));
    }
  }

  mc.create_and_finish(
      MsgContent_PaxMonStatusResponse,
      CreatePaxMonStatusResponse(
          mc, static_cast<std::uint64_t>(sched.system_time_),
          last_tick_stats_.tracked_ok_groups_ +
              last_tick_stats_.tracked_broken_groups_,
          last_tick_stats_.affected_groups_,
          last_tick_stats_.affected_passengers_,
          last_tick_stats_.broken_groups_, last_tick_stats_.broken_passengers_,
          mc.CreateVector(trips_affected_by_last_update),
          mc.CreateVector(trips_with_critical_sections))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon
