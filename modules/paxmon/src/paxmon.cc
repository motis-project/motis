#include "motis/paxmon/paxmon.h"

#include <algorithm>
#include <memory>

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
#include "motis/module/context/motis_call.h"
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
#include "motis/paxmon/messages.h"
#include "motis/paxmon/output/journey_converter.h"
#include "motis/paxmon/output/mcfp_scenario.h"
#include "motis/paxmon/over_capacity_report.h"
#include "motis/paxmon/print_stats.h"
#include "motis/paxmon/rt_updates.h"
#include "motis/paxmon/service_info.h"

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
  param(preparation_time_, "preparation_time",
        "preparation time for localization (minutes)");
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
  param(keep_group_history_, "keep_group_history",
        "keep all passenger group versions");
  param(reuse_groups_, "reuse_groups",
        "update probability of existing groups instead of adding new groups "
        "when possible");
}

paxmon::~paxmon() = default;

void paxmon::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "paxmon", reg,
      [this](event_collector::dependencies_map_t const& dependencies,
             event_collector::publish_fn_t const& /*publish*/) {
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
        LOG(info) << "paxmon: start time: "
                  << format_unix_time(start_time_.unix_time_)
                  << ", end time: " << format_unix_time(end_time_.unix_time_);

        for (auto t = start_time_.unix_time_; t <= end_time_.unix_time_;
             t += time_step_) {
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
        generate_capacities(get_sched(), data_, generated_capacity_file_);
        return {};
      });

  reg.register_op(
      "/paxmon/init_forward",
      [&](msg_ptr const&) -> msg_ptr {
        return forward(start_time_.unix_time_);
      },
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

  reg.register_op("/paxmon/get_groups", [&](msg_ptr const& msg) -> msg_ptr {
    return get_groups(msg);
  });

  reg.register_op("/paxmon/filter_groups", [&](msg_ptr const& msg) -> msg_ptr {
    return filter_groups(msg);
  });

  reg.register_op("/paxmon/filter_trips", [&](msg_ptr const& msg) -> msg_ptr {
    return filter_trips(msg);
  });

  if (!mcfp_scenario_dir_.empty()) {
    if (fs::exists(mcfp_scenario_dir_)) {
      write_mcfp_scenarios_ = fs::is_directory(mcfp_scenario_dir_);
    } else {
      write_mcfp_scenarios_ = fs::create_directories(mcfp_scenario_dir_);
    }
  }
}

loader::loader_result paxmon::load_journeys(std::string const& file) {
  auto const journey_path = fs::path{file};
  if (!fs::exists(journey_path)) {
    LOG(warn) << "journey file not found: " << file;
    return {};
  }
  auto const& sched = get_sched();
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
  auto const& sched = get_sched();
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
  auto const& sched = get_sched();
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

msg_ptr paxmon::rt_update(msg_ptr const& msg) {
  auto const& sched = get_sched();
  auto update = motis_content(RtUpdates, msg);
  handle_rt_update(data_, sched, system_stats_, tick_stats_, update,
                   arrival_delay_threshold_);
  return {};
}

void paxmon::rt_updates_applied() {
  MOTIS_START_TIMING(total);
  auto const& sched = get_sched();

  if (check_graph_times_) {
    utl::verify(check_graph_times(data_.graph_, sched),
                "rt_updates_applied: check_graph_times");
  }
  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, sched),
                "rt_updates_applied: check_graph_integrity (start)");
  }

  LOG(info) << "groups affected by last update: "
            << data_.groups_affected_by_last_update_.size()
            << ", total groups: " << data_.graph_.passenger_groups_.size();
  print_allocator_stats(data_.graph_);

  auto messages =
      update_affected_groups(data_, sched, system_stats_, tick_stats_,
                             arrival_delay_threshold_, preparation_time_);

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, sched),
                "rt_updates_applied: check_graph_integrity (after "
                "update_affected_groups)");
  }

  if (write_mcfp_scenarios_ &&
      tick_stats_.broken_groups_ >= mcfp_scenario_min_broken_groups_) {
    auto const dir =
        fs::path{mcfp_scenario_dir_} /
        fs::path{fmt::format(
            "{}_{}", format_unix_time(sched.system_time_, "%Y-%m-%d_%H-%M"),
            tick_stats_.broken_groups_)};
    LOG(info) << "writing MCFP scenario with " << tick_stats_.broken_groups_
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

  tick_stats_.t_publish_ = static_cast<std::uint64_t>(MOTIS_TIMING_MS(publish));

  tick_stats_.total_ok_groups_ = system_stats_.groups_ok_count_;
  tick_stats_.total_broken_groups_ = system_stats_.groups_broken_count_;
  tick_stats_.total_major_delay_groups_ =
      system_stats_.groups_major_delay_count_;

  LOG(info) << "affected by last rt update: "
            << data_.groups_affected_by_last_update_.size()
            << " passenger groups, "
            << " passengers";

  data_.groups_affected_by_last_update_.clear();
  LOG(info) << "passenger groups: " << tick_stats_.ok_groups_ << " ok, "
            << tick_stats_.broken_groups_
            << " broken - passengers affected by broken groups: "
            << tick_stats_.broken_passengers_;
  LOG(info) << "groups: " << system_stats_.groups_ok_count_ << " ok + "
            << system_stats_.groups_broken_count_ << " broken";

  for (auto const& pg : data_.graph_.passenger_groups_) {
    if (pg == nullptr || !pg->valid()) {
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
  auto const& sched = get_sched();
  auto const req = motis_content(PaxMonAddGroupsRequest, msg);

  auto const allow_reuse = reuse_groups_;
  auto reused_groups = 0ULL;
  auto const added_groups =
      utl::to_vec(*req->groups(), [&](PaxMonGroup const* pg_fbs) {
        utl::verify(pg_fbs->planned_journey()->legs()->size() != 0,
                    "trying to add empty passenger group");
        auto input_pg = from_fbs(sched, pg_fbs);
        if (allow_reuse) {
          if (auto it = data_.graph_.groups_by_source_.find(input_pg.source_);
              it != end(data_.graph_.groups_by_source_)) {
            for (auto const id : it->second) {
              auto existing_pg = data_.graph_.passenger_groups_.at(id);
              if (existing_pg != nullptr && existing_pg->valid() &&
                  existing_pg->compact_planned_journey_ ==
                      input_pg.compact_planned_journey_) {
                existing_pg->probability_ += input_pg.probability_;
                ++reused_groups;
                return existing_pg;
              }
            }
          }
        }
        auto pg = data_.graph_.add_group(std::move(input_pg));
        add_passenger_group_to_graph(sched, data_, *pg);
        for (auto const& leg : pg->compact_planned_journey_.legs_) {
          data_.trips_affected_by_last_update_.insert(leg.trip_);
        }
        return pg;
      });

  print_allocator_stats(data_.graph_);
  LOG(info) << "add_groups: " << added_groups.size() << " total, "
            << reused_groups << " reused";

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
    if (!keep_group_history_) {
      data_.graph_.passenger_group_allocator_.release(pg);
      pg = nullptr;
    }
  }

  print_allocator_stats(data_.graph_);

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(data_.graph_, get_sched()),
                "remove_groups (end)");
  }

  return {};
}

msg_ptr paxmon::get_trip_load_info(msg_ptr const& msg) {
  utl::verify(msg != nullptr, "null message in paxmon::get_trip_load_info");
  auto const& sched = get_sched();
  message_creator mc;

  auto const to_fbs_load_info = [&](TripId const* fbs_tid) {
    auto const trp = from_fbs(sched, fbs_tid);
    auto const tli = calc_trip_load_info(data_, trp);
    return to_fbs(mc, sched, data_.graph_, tli);
  };

  switch (msg->get()->content_type()) {
    case MsgContent_TripId: {
      auto const req = motis_content(TripId, msg);
      mc.create_and_finish(MsgContent_PaxMonTripLoadInfo,
                           to_fbs_load_info(req).Union());
      return make_msg(mc);
    }
    case MsgContent_PaxMonGetTripLoadInfosRequest: {
      auto const req = motis_content(PaxMonGetTripLoadInfosRequest, msg);
      mc.create_and_finish(
          MsgContent_PaxMonGetTripLoadInfosResponse,
          mc.CreateVector(utl::to_vec(*req->trips(), to_fbs_load_info))
              .Union());
      return make_msg(mc);
    }
    default: throw std::system_error(error::unexpected_message_type);
  }
}

msg_ptr paxmon::find_trips(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFindTripsRequest, msg);
  auto const& sched = get_sched();

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
    auto const tde = data_.graph_.trip_data_.find(trp);
    auto const has_paxmon_data = tde != end(data_.graph_.trip_data_);
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
    auto const all_edges_have_capacity_info =
        has_paxmon_data &&
        std::all_of(
            begin(tde->second->edges_), end(tde->second->edges_),
            [](edge const* e) { return !e->is_trip() || e->has_capacity(); });
    auto const has_passengers =
        has_paxmon_data &&
        std::any_of(begin(tde->second->edges_), end(tde->second->edges_),
                    [](edge const* e) {
                      return e->is_trip() &&
                             !e->get_pax_connection_info().groups_.empty();
                    });
    trips.emplace_back(CreatePaxMonTripInfo(
        mc, to_fbs_trip_service_info(mc, sched, trp, service_infos),
        has_paxmon_data, all_edges_have_capacity_info, has_passengers));
  }

  mc.create_and_finish(
      MsgContent_PaxMonFindTripsResponse,
      CreatePaxMonFindTripsResponse(mc, mc.CreateVector(trips)).Union());
  return make_msg(mc);
}

msg_ptr paxmon::get_status(msg_ptr const& /*msg*/) const {
  auto const& sched = get_sched();

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonStatusResponse,
      CreatePaxMonStatusResponse(
          mc, static_cast<std::uint64_t>(sched.system_time_),
          last_tick_stats_.tracked_ok_groups_ +
              last_tick_stats_.tracked_broken_groups_,
          last_tick_stats_.affected_groups_,
          last_tick_stats_.affected_passengers_,
          last_tick_stats_.broken_groups_, last_tick_stats_.broken_passengers_)
          .Union());
  return make_msg(mc);
}

msg_ptr paxmon::get_groups(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetGroupsRequest, msg);
  auto const& sched = get_sched();
  auto const all_generations = req->all_generations();
  auto const include_localization = req->include_localization();

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const search_time =
      static_cast<time>(current_time + req->preparation_time());
  if (include_localization) {
    utl::verify(current_time != INVALID_TIME, "invalid current system time");
  }

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonGroup>> groups;
  std::vector<flatbuffers::Offset<PaxMonLocalizationWrapper>> localizations;

  auto const add_by_data_source = [&](data_source const& ds) {
    if (auto const it = data_.graph_.groups_by_source_.find(ds);
        it != end(data_.graph_.groups_by_source_)) {
      for (auto const pgid : it->second) {
        if (auto const pg = data_.graph_.passenger_groups_.at(pgid);
            pg != nullptr) {
          if (!all_generations && !pg->valid()) {
            continue;
          }
          groups.emplace_back(to_fbs(sched, mc, *pg));
          if (include_localization) {
            localizations.emplace_back(to_fbs_localization_wrapper(
                sched, mc,
                localize(sched,
                         get_reachability(data_, pg->compact_planned_journey_),
                         search_time)));
          }
        }
      }
    }
  };

  for (auto const pgid : *req->ids()) {
    if (auto const pg = data_.graph_.passenger_groups_.at(pgid);
        pg != nullptr) {
      if (all_generations) {
        add_by_data_source(pg->source_);
      } else {
        groups.emplace_back(to_fbs(sched, mc, *pg));
      }
    }
  }

  for (auto const ds : *req->sources()) {
    add_by_data_source(from_fbs(ds));
  }

  mc.create_and_finish(
      MsgContent_PaxMonGetGroupsResponse,
      CreatePaxMonGetGroupsResponse(mc, mc.CreateVector(groups),
                                    mc.CreateVector(localizations))
          .Union());
  return make_msg(mc);
}

msg_ptr paxmon::filter_groups(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterGroupsRequest, msg);
  auto const& sched = get_sched();
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  auto const only_delayed = req->only_delayed();
  auto const min_delay = req->min_delay();
  auto const only_with_alternative_potential =
      req->only_with_alternative_potential();
  auto const preparation_time = req->preparation_time();
  auto const only_active = req->only_active();
  auto const only_original = req->only_original();
  auto const only_forecast = req->only_forecast();
  auto const include_localization = req->include_localization();

  auto const localization_needed =
      only_with_alternative_potential || include_localization;
  auto const search_time = static_cast<time>(current_time + preparation_time);

  auto total_tracked_groups = 0ULL;
  auto total_active_groups = 0ULL;
  auto filtered_original_groups = 0ULL;
  auto filtered_forecast_groups = 0ULL;
  std::vector<std::uint64_t> selected_group_ids;
  std::vector<passenger_localization> localizations;
  mcd::hash_set<data_source> selected_ds;

  for (auto const pg : data_.graph_.passenger_groups_) {
    if (pg == nullptr || (only_active && !pg->valid())) {
      continue;
    }
    ++total_tracked_groups;
    auto const est_arrival = pg->estimated_arrival_time();
    if (est_arrival != INVALID_TIME && est_arrival <= current_time) {
      continue;
    }
    ++total_active_groups;

    if (only_delayed && pg->estimated_delay() < min_delay) {
      continue;
    }

    passenger_localization localization;
    if (localization_needed) {
      auto const reachability =
          get_reachability(data_, pg->compact_planned_journey_);
      localization = localize(sched, reachability, search_time);
      if (only_with_alternative_potential &&
          localization.at_station_->index_ ==
              pg->compact_planned_journey_.destination_station_id()) {
        continue;
      }
    }

    if ((pg->source_flags_ & group_source_flags::FORECAST) ==
        group_source_flags::FORECAST) {
      if (only_original) {
        continue;
      }
      ++filtered_forecast_groups;
    } else {
      if (only_forecast) {
        continue;
      }
      ++filtered_original_groups;
    }

    selected_group_ids.emplace_back(pg->id_);
    selected_ds.insert(pg->source_);
    if (include_localization) {
      localizations.emplace_back(localization);
    }
  }

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonFilterGroupsResponse,
      CreatePaxMonFilterGroupsResponse(
          mc, total_tracked_groups, total_active_groups,
          selected_group_ids.size(), selected_ds.size(),
          filtered_original_groups, filtered_forecast_groups,
          mc.CreateVector(selected_group_ids),
          mc.CreateVector(utl::to_vec(localizations,
                                      [&](auto const& loc) {
                                        return to_fbs_localization_wrapper(
                                            sched, mc, loc);
                                      })))
          .Union());
  return make_msg(mc);
}

msg_ptr paxmon::filter_trips(msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterTripsRequest, msg);
  auto const& sched = get_sched();
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  auto const load_factor_threshold = req->load_factor_possibly_ge();
  auto const ignore_past_sections = req->ignore_past_sections();

  auto critical_sections = 0ULL;
  mcd::hash_set<trip const*> selected_trips;

  for (auto const& tde : data_.graph_.trip_data_) {
    for (auto const e : tde.second->edges_) {
      if (!e->is_trip() || !e->has_capacity()) {
        continue;
      }
      if (ignore_past_sections &&
          e->to(data_.graph_)->current_time() < current_time) {
        continue;
      }
      auto const& pci = e->get_pax_connection_info();
      auto const pdf = get_load_pdf(pci);
      auto const cdf = get_cdf(pdf);
      if (load_factor_possibly_ge(cdf, e->capacity(), load_factor_threshold)) {
        selected_trips.insert(tde.first);
        ++critical_sections;
      }
    }
  }

  message_creator mc;
  auto const selected_tsis = utl::to_vec(selected_trips, [&](trip const* trp) {
    return to_fbs_trip_service_info(mc, sched, trp);
  });

  mc.create_and_finish(MsgContent_PaxMonFilterTripsResponse,
                       CreatePaxMonFilterTripsResponse(
                           mc, selected_trips.size(), critical_sections,
                           mc.CreateVector(selected_tsis))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon
