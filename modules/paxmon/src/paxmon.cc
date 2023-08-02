#include "motis/paxmon/paxmon.h"

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <memory>

#include "fmt/format.h"

#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/event_collector.h"
#include "motis/module/message.h"

#include "motis/paxmon/api/add_groups.h"
#include "motis/paxmon/api/capacity_status.h"
#include "motis/paxmon/api/debug_graph.h"
#include "motis/paxmon/api/destroy_universe.h"
#include "motis/paxmon/api/detailed_capacity_status.h"
#include "motis/paxmon/api/filter_groups.h"
#include "motis/paxmon/api/filter_trips.h"
#include "motis/paxmon/api/find_trips.h"
#include "motis/paxmon/api/fork_universe.h"
#include "motis/paxmon/api/get_addressable_groups.h"
#include "motis/paxmon/api/get_groups.h"
#include "motis/paxmon/api/get_groups_in_trip.h"
#include "motis/paxmon/api/get_interchanges.h"
#include "motis/paxmon/api/get_status.h"
#include "motis/paxmon/api/get_trip_capacity.h"
#include "motis/paxmon/api/get_trip_load_info.h"
#include "motis/paxmon/api/get_universes.h"
#include "motis/paxmon/api/group_statistics.h"
#include "motis/paxmon/api/keep_alive.h"
#include "motis/paxmon/api/metrics.h"
#include "motis/paxmon/api/remove_groups.h"
#include "motis/paxmon/api/reroute_groups.h"

#include "motis/paxmon/broken_interchanges_report.h"
#include "motis/paxmon/checks.h"
#include "motis/paxmon/generate_capacities.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/loader/capacities/load_capacities.h"
#include "motis/paxmon/loader/csv_journeys/csv_journeys.h"
#include "motis/paxmon/loader/dailytrek.h"
#include "motis/paxmon/loader/motis_journeys/motis_journeys.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/output/journey_converter.h"
#include "motis/paxmon/output/mcfp_scenario.h"
#include "motis/paxmon/over_capacity_report.h"
#include "motis/paxmon/print_stats.h"
#include "motis/paxmon/rt_updates.h"
#include "motis/paxmon/service_info.h"
#include "motis/paxmon/tools/commands.h"

namespace fs = std::filesystem;

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;

namespace motis::paxmon {

paxmon::paxmon() : module("Passenger Monitoring", "paxmon"), data_{*this} {
  param(journey_input_settings_.journey_timezone_, "journey_timezone",
        "timezone for timestamps in daily trek input files (e.g. "
        "Europe/Berlin), or empty for current system time zone");
  param(journey_input_settings_.journey_match_log_file_, "journey_match_log",
        "journey match log file");
  param(journey_input_settings_.match_tolerance_, "match_tolerance",
        "journey match time tolerance (minutes)");
  param(journey_input_settings_.split_groups_, "split_groups",
        "split groups from journey input files into smaller groups");
  param(journey_input_settings_.split_groups_size_mean_, "split_groups_mean",
        "mean size for split groups");
  param(journey_input_settings_.split_groups_size_stddev_,
        "split_groups_stddev", "standard deviation for split groups size");
  param(journey_input_settings_.split_groups_seed_, "split_groups_seed",
        "rng seed for splitting groups");
  param(journey_input_settings_.max_station_wait_time_, "max_station_wait_time",
        "maximum wait time at a station, if exceeded the journey is split into "
        "separate journeys (minutes, set to 0 to disable)");

  param(generated_capacity_file_, "generated_capacity_file",
        "output for generated capacities");
  param(stats_file_, "stats", "statistics file");
  param(capacity_match_log_file_, "capacity_match_log",
        "capacity match log file");
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
  param(graph_log_enabled_, "graph_log", "enable graph log");
  param(capacity_fuzzy_match_max_time_diff_,
        "capacity_fuzzy_match_max_time_diff",
        "max allowed departure/arrival time difference in minutes for fuzzy "
        "matching trip capacity data (0 to disable fuzzy matching)");
  param(min_capacity_, "min_capacity",
        "minimum capacity override (if capacity data is available but lower "
        "than this value, the minimum is used)");
}

paxmon::~paxmon() = default;

void paxmon::reg_subc(motis::module::subc_reg& r) {
  r.register_cmd("paxmon_convert", "convert journeys to csv", tools::convert);
  r.register_cmd("paxmon_generate", "generate journeys", tools::generate);
  r.register_cmd("paxmon_groups", "generate groups", tools::gen_groups);
}

void paxmon::import(motis::module::import_dispatcher& reg) {
  add_shared_data(to_res_id(global_res_id::PAX_DATA), &data_);
  auto* uv = data_.multiverse_->create_default_universe();
  uv->graph_log_.enabled_ = graph_log_enabled_;
  uv->capacity_maps_.fuzzy_match_max_time_diff_ =
      capacity_fuzzy_match_max_time_diff_;
  uv->capacity_maps_.min_capacity_ = min_capacity_;

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
            } else if (fs::is_directory(path)) {
              journey_dirs_.emplace_back(path);
            } else {
              LOG(warn) << "journey file not found: " << path;
              import_successful_ = false;
            }
          }
        }

        load_capacity_files();
        find_journey_files();
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

  reg.subscribe(
      "/init",
      [&]() {
        auto const& primary_uv = primary_universe();
        if (primary_uv.capacity_maps_.trip_capacity_map_.empty() &&
            primary_uv.capacity_maps_.category_capacity_map_.empty()) {
          LOG(warn) << "no capacity information available";
        }
        LOG(info) << "tracking " << primary_uv.passenger_groups_.size()
                  << " passenger groups";

        shared_data_->register_timer("PaxMon Universe GC",
                                     boost::posix_time::seconds{10},
                                     [this]() { universe_gc(); }, {});
      },
      {ctx::access_request{to_res_id(global_res_id::SCHEDULE),
                           ctx::access_t::READ},
       ctx::access_request{to_res_id(global_res_id::PAX_DEFAULT_UNIVERSE),
                           ctx::access_t::READ}});

  reg.register_op("/paxmon/flush",
                  [&](msg_ptr const&) -> msg_ptr {
                    stats_writer_->flush();
                    return {};
                  },
                  {});

  reg.subscribe("/rt/update",
                [&](msg_ptr const& msg) { return rt_update(msg); }, {});

  reg.subscribe("/rt/graph_updated",
                [&](msg_ptr const& msg) {
                  rt_updates_applied(msg);
                  if (!initial_forward_done_ &&
                      motis_content(RtGraphUpdated, msg)->schedule() == 0U) {
                    initial_forward_done_ = true;
                    motis_call(make_no_msg("/paxmon/init_forward"))->val();
                  }
                  return nullptr;
                },
                {});

  auto const forward = [](std::time_t time) {
    using namespace motis::ris;
    message_creator fbb;
    fbb.create_and_finish(MsgContent_RISForwardTimeRequest,
                          CreateRISForwardTimeRequest(fbb, time).Union(),
                          "/ris/forward");
    LOG(info) << "paxmon: forwarding time to: " << format_unix_time(time)
              << " (" << time << ") =========================================";
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
      {ctx::access_request{to_res_id(global_res_id::SCHEDULE),
                           ctx::access_t::WRITE},
       ctx::access_request{to_res_id(global_res_id::PAX_DEFAULT_UNIVERSE),
                           ctx::access_t::WRITE}});

  // --init /paxmon/generate_capacities
  // --paxmon.generated_capacity_file file.csv
  reg.register_op(
      "/paxmon/generate_capacities",
      [&](msg_ptr const&) -> msg_ptr {
        if (generated_capacity_file_.empty()) {
          LOG(logging::error)
              << "generate_capacities: no output file specified";
          return {};
        }
        generate_capacities(get_sched(), primary_universe(),
                            generated_capacity_file_);
        return {};
      },
      {ctx::access_request{to_res_id(global_res_id::SCHEDULE),
                           ctx::access_t::READ},
       ctx::access_request{to_res_id(global_res_id::PAX_DEFAULT_UNIVERSE),
                           ctx::access_t::READ}});

  reg.register_op(
      "/paxmon/init_forward",
      [&](msg_ptr const&) -> msg_ptr {
        auto const& sched = get_sched();
        if (start_time_.unix_time_ != 0 &&
            start_time_.unix_time_ > sched.system_time_ && time_step_ != 0) {
          LOG(info) << "paxmon: forwarding time: "
                    << format_unix_time(sched.system_time_) << " -> "
                    << format_unix_time(start_time_.unix_time_) << " in "
                    << time_step_ << "s intervals";
          for (auto t = sched.system_time_ + time_step_;
               t <= start_time_.unix_time_; t += time_step_) {
            forward(t);
          }
        }
        return {};
      },
      {ctx::access_request{to_res_id(global_res_id::SCHEDULE),
                           ctx::access_t::WRITE},
       ctx::access_request{to_res_id(global_res_id::PAX_DEFAULT_UNIVERSE),
                           ctx::access_t::WRITE}});

  reg.register_op("/paxmon/add_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::add_groups(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/remove_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::remove_groups(data_, check_graph_integrity_,
                                              msg);
                  },
                  {});

  reg.register_op("/paxmon/reroute_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::reroute_groups(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/trip_load_info",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_trip_load_info(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/groups_in_trip",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_groups_in_trip(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/trip_capacity",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_trip_capacity(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/addressable_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_addressable_groups(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/find_trips",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::find_trips(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/status",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_status(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/universes",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_universes(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/get_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_groups(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/filter_groups",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::filter_groups(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/group_statistics",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::group_statistics(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/filter_trips",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::filter_trips(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/get_interchanges",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::get_interchanges(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/fork_universe",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::fork_universe(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/destroy_universe",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::destroy_universe(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/keep_alive",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::keep_alive(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/debug_graph",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::debug_graph(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/capacity_status",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::capacity_status(data_, msg);
                  },
                  {});

  reg.register_op("/paxmon/detailed_capacity_status",
                  [&](msg_ptr const& msg) -> msg_ptr {
                    return api::detailed_capacity_status(data_, msg);
                  },
                  {});

  reg.register_op(
      "/paxmon/metrics",
      [&](msg_ptr const& msg) -> msg_ptr { return api::metrics(data_, msg); },
      {});

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
  auto& uv = primary_universe();
  auto result = loader::loader_result{};
  if (journey_path.extension() == ".txt") {
    scoped_timer const journey_timer{"load motis journeys"};
    result = loader::motis_journeys::load_journeys(sched, uv, file);
  } else if (journey_path.extension() == ".csv") {
    scoped_timer const journey_timer{"load csv journeys"};
    result = loader::csv_journeys::load_journeys(sched, uv, file,
                                                 journey_input_settings_);
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
  auto& uv = primary_universe();
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Load Journeys")
      .out_bounds(10.F, 100.F)
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
        scoped_timer const timer{"reroute unmatched journeys"};
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
          loader::motis_journeys::load_journey(
              sched, uv, journeys.front(), uj.source_, uj.passengers_,
              route_source_flags::MATCH_REROUTED);
        }
      }
      progress_tracker->increment();
    }
  }

  {
    scoped_timer const timer{"init expected load"};
    uv.pax_connection_info_.init_expected_load(uv.passenger_groups_);
  }

  auto const graph_stats = calc_graph_statistics(sched, uv);
  print_graph_stats(graph_stats);
  print_allocator_stats(uv);
  if (graph_stats.trips_over_capacity_ > 0 &&
      !initial_over_capacity_report_file_.empty()) {
    write_over_capacity_report(uv, sched, initial_over_capacity_report_file_);
  }
  if (!initial_broken_report_file_.empty()) {
    write_broken_interchanges_report(uv, initial_broken_report_file_);
  }

  if (check_graph_times_) {
    utl::verify(check_graph_times(uv, sched),
                "load_journeys: check_graph_times");
  }
  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(uv, sched),
                "load_journeys: check_graph_integrity");
  }
}

void paxmon::find_journey_files() {
  auto const& sched = get_sched();

  for (auto const& dir : journey_dirs_) {
    auto const files = loader::get_dailytrek_files(sched, dir);
    std::copy(begin(files), end(files), std::back_inserter(journey_files_));
  }
}

void paxmon::load_capacity_files() {
  auto const& sched = get_sched();
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Load Capacity Data")
      .out_bounds(0.F, 10.F)
      .in_high(capacity_files_.size());
  auto total_entries = 0ULL;
  auto& primary_uv = primary_universe();
  for (auto const& file : capacity_files_) {
    auto const capacity_path = fs::path{file};
    if (!fs::exists(capacity_path)) {
      LOG(warn) << "capacity file not found: " << file;
      import_successful_ = false;
      continue;
    }
    auto const res = loader::capacities::load_capacities_from_file(
        sched, primary_uv.capacity_maps_, file, capacity_match_log_file_);
    total_entries += res.loaded_entry_count_;
    LOG(info) << fmt::format("loaded {:L} capacity entries from {}",
                             res.loaded_entry_count_, file);
    progress_tracker->increment();
  }
  if (total_entries == 0) {
    LOG(warn)
        << "no capacity data loaded, all trips will have unknown capacity";
  }
}

// called after rt propagate
msg_ptr paxmon::rt_update(msg_ptr const& msg) {
  auto const update = motis_content(RtUpdates, msg);
  auto const schedule_res_id = update->schedule();
  auto const uv_ids =
      data_.multiverse_->universes_using_schedule(schedule_res_id);
  for (auto const uv_id : uv_ids) {
    auto const uv_access =
        get_universe_and_schedule(data_, uv_id, ctx::access_t::WRITE);
    handle_rt_update(uv_access.uv_, uv_access.sched_, update,
                     arrival_delay_threshold_);
  }
  return {};
}

// called after rt flush
void paxmon::rt_updates_applied(msg_ptr const& msg) {
  scoped_timer const t{"paxmon: graph_updated"};
  auto const rgu = motis_content(RtGraphUpdated, msg);
  auto const schedule_res_id = rgu->schedule();
  auto const uv_ids =
      data_.multiverse_->universes_using_schedule(schedule_res_id);
  for (auto const uv_id : uv_ids) {
    auto const uv_access =
        get_universe_and_schedule(data_, uv_id, ctx::access_t::WRITE);
    rt_updates_applied(uv_access.uv_, uv_access.sched_);
  }
}

void paxmon::rt_updates_applied(universe& uv, schedule const& sched) {
  MOTIS_START_TIMING(total);
  if (check_graph_times_) {
    utl::verify(check_graph_times(uv, sched),
                "rt_updates_applied: check_graph_times");
  }
  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(uv, sched),
                "rt_updates_applied: check_graph_integrity (start)");
  }

  LOG(info) << "group routes affected by last update: "
            << uv.rt_update_ctx_.group_routes_affected_by_last_update_.size();
  print_allocator_stats(uv);

  auto messages = update_affected_groups(uv, sched, arrival_delay_threshold_,
                                         preparation_time_);

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(uv, sched),
                "rt_updates_applied: check_graph_integrity (after "
                "update_affected_groups)");
  }

  if (write_mcfp_scenarios_ &&
      uv.tick_stats_.broken_group_routes_ >= mcfp_scenario_min_broken_groups_) {
    auto const dir =
        fs::path{mcfp_scenario_dir_} /
        fs::path{fmt::format(
            "{}_{}", format_unix_time(sched.system_time_, "%Y-%m-%d_%H-%M"),
            uv.tick_stats_.broken_group_routes_)};
    LOG(info) << "writing MCFP scenario with "
              << uv.tick_stats_.broken_group_routes_
              << " broken group routes to " << dir.string();
    fs::create_directories(dir);
    output::write_scenario(dir, sched, uv, messages,
                           mcfp_scenario_include_trip_info_);
  }

  MOTIS_START_TIMING(publish);
  for (auto& msg : messages) {
    ctx::await_all(motis_publish(msg));
    msg.reset();
  }
  MOTIS_STOP_TIMING(publish);

  uv.tick_stats_.t_publish_ =
      static_cast<std::uint64_t>(MOTIS_TIMING_MS(publish));

  LOG(info) << "affected by last rt update: "
            << uv.rt_update_ctx_.group_routes_affected_by_last_update_.size()
            << " passenger group routes";

  uv.rt_update_ctx_.group_routes_affected_by_last_update_.clear();
  LOG(info) << "passenger group routes: " << uv.tick_stats_.ok_group_routes_
            << " ok, " << uv.tick_stats_.broken_group_routes_ << " broken";

  MOTIS_STOP_TIMING(total);
  uv.tick_stats_.t_rt_updates_applied_total_ =
      static_cast<std::uint64_t>(MOTIS_TIMING_MS(total));

  uv.metrics_.add(sched.system_time_, now(), uv.tick_stats_);
  uv.update_tracker_.rt_updates_applied(uv.tick_stats_);
  if (uv.id_ == 0) {
    stats_writer_->write_tick(uv.tick_stats_);
    stats_writer_->flush();
  }
  uv.tick_stats_ = {};

  if (check_graph_integrity_) {
    utl::verify(check_graph_integrity(uv, sched),
                "rt_updates_applied: check_graph_integrity (end)");
  }
}

void paxmon::universe_gc() const {
  data_.multiverse_->destroy_expired_universes();
}

universe& paxmon::primary_universe() {
  return *get_shared_data<std::unique_ptr<universe>>(motis::module::to_res_id(
      motis::module::global_res_id::PAX_DEFAULT_UNIVERSE));
}

}  // namespace motis::paxmon
