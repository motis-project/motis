#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"

#include "utl/progress_tracker.h"
#include "utl/raii.h"
#include "utl/to_vec.h"

#include "motis/tripbased/data.h"
#include "motis/tripbased/debug.h"
#include "motis/tripbased/error.h"
#include "motis/tripbased/lower_bounds.h"
#include "motis/tripbased/preprocessing.h"
#include "motis/tripbased/query.h"
#include "motis/tripbased/tb_journey.h"
#include "motis/tripbased/tb_ontrip_search.h"
#include "motis/tripbased/tb_profile_search.h"
#include "motis/tripbased/tb_to_journey.h"
#include "motis/tripbased/tripbased.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/statistics/statistics.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::routing;
using namespace flatbuffers;

namespace motis::tripbased {

inline bool is_virtual_start_station(station_id id) { return id == 0; }

inline bool is_virtual_end_station(station_id id) { return id == 1; }

inline station_node const* get_station_node(schedule const& sched,
                                            InputStation const* input_station) {
  using guesser::StationGuesserResponse;

  std::string station_id;

  if (input_station->id()->Length() != 0) {
    station_id = input_station->id()->str();
  } else {
    module::message_creator b;
    b.create_and_finish(MsgContent_StationGuesserRequest,
                        guesser::CreateStationGuesserRequest(
                            b, 1, b.CreateString(input_station->name()->str()))
                            .Union(),
                        "/guesser");
    auto const msg = motis_call(make_msg(b))->val();
    auto const guesses = motis_content(StationGuesserResponse, msg)->guesses();
    if (guesses->size() == 0) {
      throw std::system_error(error::no_guess_for_station);
    }
    station_id = guesses->Get(0)->id()->str();
  }

  return motis::get_station_node(sched, station_id);
}

struct trip_based_result {
  trip_based_result() = default;
  explicit trip_based_result(std::vector<stats_category>&& stats)
      : stats_(std::move(stats)) {}

  std::vector<journey> journeys_;
  std::vector<stats_category> stats_;
  time interval_begin_{};
  time interval_end_{};
};

trip_based_query build_tb_query(RoutingRequest const* req,
                                schedule const& sched) {
  trip_based_query q;
  q.start_type_ = req->start_type();

  if (req->search_type() != SearchType_Default &&
      req->search_type() != SearchType_Accessibility) {
    throw std::system_error(error::search_type_not_supported);
  }

  if (req->via()->size() != 0) {
    throw std::system_error(error::via_not_supported);
  }

  if (req->include_equivalent()) {
    throw std::system_error(error::include_equivalent_not_supported);
  }

  for (auto const& aew : *req->additional_edges()) {
    if (aew->additional_edge_type() != AdditionalEdge_MumoEdge) {
      throw std::system_error(error::invalid_additional_edges);
    }
  }

  std::string start_node_eva;

  switch (q.start_type_) {
    case Start_OntripStationStart: {
      auto const start =
          reinterpret_cast<OntripStationStart const*>(req->start());
      verify_external_timestamp(sched, start->departure_time());

      auto const start_node = get_station_node(sched, start->station());
      start_node_eva = sched.stations_[start_node->id_]->eva_nr_;

      q.start_station_ = start_node->id_;
      q.start_time_ = unix_to_motistime(sched, start->departure_time());
      break;
    }
    case Start_PretripStart: {
      auto const start = reinterpret_cast<PretripStart const*>(req->start());
      verify_external_timestamp(sched, start->interval()->begin());
      verify_external_timestamp(sched, start->interval()->end());

      auto const start_node = get_station_node(sched, start->station());
      start_node_eva = sched.stations_[start_node->id_]->eva_nr_;

      q.start_station_ = start_node->id_;
      q.interval_begin_ = unix_to_motistime(sched, start->interval()->begin());
      q.interval_end_ = unix_to_motistime(sched, start->interval()->end());
      q.extend_interval_earlier_ = start->extend_interval_earlier();
      q.extend_interval_later_ = start->extend_interval_later();
      q.min_connection_count_ = start->min_connection_count();
      break;
    }
    default: throw std::system_error(error::start_type_not_supported);
  }

  auto const destination_node = get_station_node(sched, req->destination());
  auto const destination_node_eva =
      sched.stations_[destination_node->id_]->eva_nr_;

  q.destination_station_ = destination_node->id_;
  q.dir_ = req->search_dir() == SearchDir_Forward ? search_dir::FWD
                                                  : search_dir::BWD;

  q.intermodal_start_ = is_virtual_start_station(q.start_station_) ||
                        is_virtual_end_station(q.start_station_);
  q.intermodal_destination_ =
      is_virtual_start_station(q.destination_station_) ||
      is_virtual_end_station(q.destination_station_);

  q.use_start_metas_ = req->use_start_metas() && !q.intermodal_start_ &&
                       q.start_type_ == Start_PretripStart;
  q.use_dest_metas_ = req->use_dest_metas() && !q.intermodal_destination_;
  q.use_start_footpaths_ = req->use_start_footpaths();

  if (q.use_start_metas_) {
    q.meta_starts_ = utl::to_vec(sched.stations_[q.start_station_]->equivalent_,
                                 [](station const* st) { return st->index_; });
  } else {
    q.meta_starts_.push_back(q.start_station_);
  }

  if (q.use_dest_metas_) {
    q.meta_destinations_ =
        utl::to_vec(sched.stations_[q.destination_station_]->equivalent_,
                    [](station const* st) { return st->index_; });
  } else {
    q.meta_destinations_.push_back(q.destination_station_);
  }

  for (auto const& aew : *req->additional_edges()) {
    assert(aew->additional_edge_type() == AdditionalEdge_MumoEdge);
    auto const info = reinterpret_cast<MumoEdge const*>(aew->additional_edge());
    if (q.dir_ == search_dir::FWD) {
      if (start_node_eva == info->from_station_id()->str()) {
        auto const st =
            get_station(sched, info->to_station_id()->str())->index_;
        q.start_edges_.emplace_back(st, info);
      } else if (destination_node_eva == info->to_station_id()->str()) {
        auto const st =
            get_station(sched, info->from_station_id()->str())->index_;
        q.destination_edges_.emplace_back(st, info);
      }
    } else {
      if (start_node_eva == info->to_station_id()->str()) {
        auto const st =
            get_station(sched, info->from_station_id()->str())->index_;
        q.start_edges_.emplace_back(st, info);
      } else if (destination_node_eva == info->from_station_id()->str()) {
        auto const st =
            get_station(sched, info->to_station_id()->str())->index_;
        q.destination_edges_.emplace_back(st, info);
      }
    }
  }

  return q;
}

inline bool incomparable(tb_journey const& a, tb_journey const& b,
                         bool pretrip) {
  return pretrip && (a.actual_departure_time() < b.actual_departure_time() ||
                     a.actual_arrival_time() > b.actual_arrival_time());
}

inline bool dominates(tb_journey const& a, tb_journey const& b, bool pretrip) {
  if (incomparable(a, b, pretrip)) {
    return false;
  }
  return (pretrip ? a.actual_duration() <= b.actual_duration()
                  : a.duration() <= b.duration()) &&
         a.transfers_ <= b.transfers_;
}

void add_result(std::vector<tb_journey>& results, tb_journey const& tbj,
                trip_based_query const& q) {
  if (tbj.actual_duration() > MAX_TRAVEL_TIME) {
    return;
  }
  auto const pretrip = q.is_pretrip();
  if (std::any_of(begin(results), end(results),
                  [&](tb_journey const& existing) {
                    return dominates(existing, tbj, pretrip);
                  })) {
    return;
  }
  utl::erase_if(results, [&](tb_journey const& existing) {
    return dominates(tbj, existing, pretrip);
  });
  results.push_back(tbj);
}

struct tripbased::impl {
  explicit impl(std::unique_ptr<tb_data> data) : tb_data_{std::move(data)} {}

  msg_ptr route(msg_ptr const& msg) {
    MOTIS_START_TIMING(total_timing);
    auto const req = motis_content(RoutingRequest, msg);

    auto const& sched = get_schedule();
    auto const query = build_tb_query(req, sched);

    auto res = route_dispatch(query, sched);

    MOTIS_STOP_TIMING(total_timing);

    message_creator fbb;
    auto stats =
        utl::to_vec(res.stats_, [&](auto const& s) { return to_fbs(fbb, s); });
    fbb.create_and_finish(
        MsgContent_RoutingResponse,
        CreateRoutingResponse(
            fbb, fbb.CreateVectorOfSortedTables(&stats),
            fbb.CreateVector(utl::to_vec(
                res.journeys_,
                [&](journey const& j) { return to_connection(fbb, j); })),
            static_cast<uint64_t>(
                motis_to_unixtime(sched, res.interval_begin_)),
            static_cast<uint64_t>(motis_to_unixtime(sched, res.interval_end_)),
            fbb.CreateVector(std::vector<Offset<DirectConnection>>{}))
            .Union());
    return make_msg(fbb);
  }

  inline trip_based_result route_dispatch(trip_based_query const& q,
                                          schedule const& sched) {
    if ((q.intermodal_start_ && q.start_edges_.empty()) ||
        (q.intermodal_destination_ && q.destination_edges_.empty())) {
      return {};
    }
    if (q.dir_ == search_dir::FWD) {
      return route_dispatch_dir<search_dir::FWD>(q, sched);
    } else {
      return route_dispatch_dir<search_dir::BWD>(q, sched);
    }
  }

  template <search_dir Dir>
  inline trip_based_result route_dispatch_dir(trip_based_query const& q,
                                              schedule const& sched) {
    if (q.is_ontrip()) {
      return route_ontrip_station<Dir>(q, sched);
    } else {
      return route_pretrip<Dir>(q, sched);
    }
  }

  template <search_dir Dir>
  trip_based_result route_ontrip_station(trip_based_query const& q,
                                         schedule const& sched) {
    trip_based_result res{};
    MOTIS_START_TIMING(search_timing);
    tb_ontrip_search<Dir> tbs(
        *tb_data_, sched, q.start_time_, q.intermodal_start_,
        q.intermodal_destination_,
        q.use_dest_metas_ ? destination_mode::ANY : destination_mode::ALL);

    add_starts_and_destinations(q, tbs);

    tbs.search();
    MOTIS_STOP_TIMING(search_timing);
    tbs.get_statistics().search_duration_ = MOTIS_TIMING_MS(search_timing);

    res.stats_.emplace_back(
        to_stats_category("tripbased", tbs.get_statistics()));

    build_results<Dir>(q, res, sched, tbs);
    return res;
  }

  template <search_dir Dir>
  trip_based_result route_pretrip(trip_based_query const& q,
                                  schedule const& sched) {
    trip_based_result res{};
    std::vector<tb_statistics> tb_stats;
    MOTIS_START_TIMING(total_search_timing);
    auto max_interval_reached = false;
    auto interval_begin = q.interval_begin_;
    auto interval_end = q.interval_end_;
    uint64_t interval_extensions = 0;
    auto const schedule_begin = SCHEDULE_OFFSET_MINUTES;
    auto const schedule_end =
        static_cast<time>((sched.schedule_end_ - sched.schedule_begin_) / 60);
    uint64_t lower_bounds_duration = 0;

    auto const map_to_interval = [&schedule_begin, &schedule_end](time t) {
      return std::min(schedule_end, std::max(schedule_begin, t));
    };

    auto const extend_interval = [&](duration const extension) {
      interval_begin =
          q.extend_interval_earlier_
              ? map_to_interval(static_cast<time>(interval_begin - extension))
              : interval_begin;
      interval_end =
          q.extend_interval_later_
              ? map_to_interval(static_cast<time>(interval_end + extension))
              : interval_end;
    };

    auto const extended_initial_interval =
        q.min_connection_count_ > 0 &&
        (q.extend_interval_earlier_ || q.extend_interval_later_) &&
        (q.interval_end_ - q.interval_begin_) < 8 * 60;
    if (extended_initial_interval) {
      extend_interval(120);
    }

    auto const in_interval = [&sched](journey const& j,
                                      time const test_interval_begin,
                                      time const test_interval_end) {
      auto const ts = unix_to_motistime(
          sched, Dir == search_dir::FWD
                     ? j.stops_.front().departure_.schedule_timestamp_
                     : j.stops_.back().arrival_.schedule_timestamp_);
      return ts >= test_interval_begin && ts <= test_interval_end;
    };

    auto const number_of_results_in_interval =
        [&](time const test_interval_begin, time const test_interval_end) {
          return std::count_if(
              begin(res.journeys_), end(res.journeys_), [&](journey const& j) {
                return in_interval(j, test_interval_begin, test_interval_end);
              });
        };

    auto const dest_mode =
        q.use_dest_metas_ ? destination_mode::ANY : destination_mode::ALL;

    while (!max_interval_reached) {
      MOTIS_START_TIMING(iteration_search_timing);
      max_interval_reached =
          (!q.extend_interval_earlier_ || interval_begin == schedule_begin) &&
          (!q.extend_interval_later_ || interval_end == schedule_end);

      tb_profile_search<Dir> tbs(*tb_data_, sched, interval_begin, interval_end,
                                 q.intermodal_start_, q.intermodal_destination_,
                                 dest_mode);
      add_starts_and_destinations(q, tbs);
      tbs.search();
      res.interval_begin_ = interval_begin;
      res.interval_end_ = interval_end;
      build_results<Dir>(q, res, sched, tbs);
      MOTIS_STOP_TIMING(iteration_search_timing);
      tbs.get_statistics().search_duration_ =
          MOTIS_TIMING_MS(iteration_search_timing);
      tb_stats.push_back(tbs.get_statistics());
      if (res.journeys_.size() >= q.min_connection_count_) {
        break;
      }

      if (interval_extensions == 0 && res.journeys_.empty()) {
        if (!is_reachable(sched, q, lower_bounds_duration)) {
          break;
        }
      }

      if (!max_interval_reached) {
        extend_interval(120);
        res.journeys_.clear();
        ++interval_extensions;
      }
    }
    MOTIS_STOP_TIMING(total_search_timing);

    uint64_t results_outside_of_interval = 0;
    auto const results_in_query_interval =
        number_of_results_in_interval(q.interval_begin_, q.interval_end_);
    if (extended_initial_interval &&
        results_in_query_interval >= q.min_connection_count_) {
      auto const results_before = res.journeys_.size();
      utl::erase_if(res.journeys_, [&](journey const& j) {
        return !in_interval(j, q.interval_begin_, q.interval_end_);
      });
      results_outside_of_interval = results_before - res.journeys_.size();
    }

    for (auto& tbs : tb_stats) {
      tbs.lower_bounds_duration_ = lower_bounds_duration;
    }

    res.stats_.emplace_back(to_stats_category("tripbased", tb_stats.back()));
    res.stats_.emplace_back(stats_category{
        "tripbased.pretrip",
        {{"interval_extensions", interval_extensions},
         {"total_search_duration",
          static_cast<uint64_t>(MOTIS_TIMING_MS(total_search_timing))},
         {"max_interval_reached", static_cast<uint64_t>(max_interval_reached)},
         {"extended_initial_interval",
          static_cast<uint64_t>(extended_initial_interval)},
         {"results_outside_of_interval", results_outside_of_interval},
         {"results_in_query_interval",
          static_cast<uint64_t>(results_in_query_interval)}}});

    return res;
  }

  static bool is_reachable(schedule const& sched, trip_based_query const& q,
                           uint64_t& lower_bounds_duration) {
    MOTIS_START_TIMING(lower_bounds_timing);
    auto const lbs = calc_lower_bounds(sched, q);
    MOTIS_STOP_TIMING(lower_bounds_timing);
    lower_bounds_duration =
        static_cast<uint64_t>(MOTIS_TIMING_MS(lower_bounds_timing));
    return std::any_of(
        begin(q.meta_starts_), end(q.meta_starts_),
        [&](station_id const station) {
          return lbs.travel_time_.is_reachable(
              lbs.travel_time_[sched.station_nodes_[station].get()]);
        });
  }

  template <typename TBS>
  void add_starts_and_destinations(trip_based_query const& q, TBS& tbs) {
    if (!q.intermodal_start_) {
      for (auto const& station : q.meta_starts_) {
        tbs.add_start(station, 0, q.use_start_footpaths_);
      }
    }

    if (!q.intermodal_destination_) {
      for (auto const& station : q.meta_destinations_) {
        tbs.add_destination(station, true);
      }
    }

    for (auto const& e : q.start_edges_) {
      tbs.add_start(e.station_id_, e.duration_, q.use_start_footpaths_);
    }

    for (auto const& e : q.destination_edges_) {
      tbs.add_destination(e.station_id_, true);
    }
  }

  template <search_dir Dir, typename TBS>
  void build_results(trip_based_query const& q, trip_based_result& res,
                     schedule const& sched, TBS& tbs) {
    std::vector<tb_journey> results;
    auto const add_starts = [&](std::vector<tb_journey>& tbjs) {
      for (auto& tbj : tbjs) {
        if (tbj.edges_.empty()) {
          LOG(warn) << "tripbased journey has no edges!";
          continue;
        }
        if (q.intermodal_start_) {
          if (Dir == search_dir::FWD) {
            auto const enter_station = tbj.start_station_;
            for (auto const& e : q.start_edges_) {
              if (e.station_id_ != enter_station) {
                continue;
              }
              auto const& first_edge = tbj.edges_.front();
              auto connection_departure = tbj.edges_.front().departure_time_;
              if (first_edge.is_connection()) {
                connection_departure -=
                    sched.stations_[e.station_id_]->transfer_time_;
              }
              auto j = tbj;
              j.edges_.emplace(
                  begin(j.edges_),
                  tb_footpath{q.start_station_, e.station_id_, e.duration_},
                  connection_departure - e.duration_, connection_departure,
                  e.mumo_id_, e.price_, e.accessibility_);
              add_result(results, j, q);
            }
          } else {
            auto const enter_station = tbj.start_station_;
            for (auto const& e : q.start_edges_) {
              if (e.station_id_ != enter_station) {
                continue;
              }
              auto const& last_edge = tbj.edges_.back();
              auto connection_arrival = last_edge.arrival_time_;
              if (last_edge.is_connection()) {
                connection_arrival +=
                    sched.stations_[e.station_id_]->transfer_time_;
              }
              auto j = tbj;
              j.edges_.emplace_back(
                  tb_footpath{e.station_id_, q.start_station_, e.duration_},
                  connection_arrival, connection_arrival + e.duration_,
                  e.mumo_id_, e.price_, e.accessibility_);
              add_result(results, j, q);
            }
          }
        } else {
          add_result(results, tbj, q);
        }
      }
    };

    if (q.intermodal_destination_) {
      for (auto const& e : q.destination_edges_) {
        auto tbjs = tbs.get_results(e.station_id_);
        for (auto& tbj : tbjs) {
          if (tbj.edges_.empty()) {
            continue;
          }
          if (Dir == search_dir::FWD) {
            auto const& last_edge = tbj.edges_.back();
            auto connection_arrival = last_edge.arrival_time_;
            if (last_edge.is_connection()) {
              connection_arrival +=
                  sched.stations_[e.station_id_]->transfer_time_;
            }
            tbj.edges_.emplace_back(
                tb_footpath{e.station_id_, q.destination_station_, e.duration_},
                connection_arrival, connection_arrival + e.duration_,
                e.mumo_id_, e.price_, e.accessibility_);
          } else {
            auto const& first_edge = tbj.edges_.front();
            auto connection_departure = tbj.edges_.front().departure_time_;
            if (first_edge.is_connection()) {
              connection_departure -=
                  sched.stations_[e.station_id_]->transfer_time_;
            }
            tbj.edges_.emplace(
                begin(tbj.edges_),
                tb_footpath{q.destination_station_, e.station_id_, e.duration_},
                connection_departure - e.duration_, connection_departure,
                e.mumo_id_, e.price_, e.accessibility_);
          }
        }
        add_starts(tbjs);
      }
    } else {
      for (auto const& dest_station : q.meta_destinations_) {
        auto tbjs = tbs.get_results(dest_station);
        add_starts(tbjs);
      }
    }

    filter_results(results, q, res);

    res.journeys_ = utl::to_vec(results, [&](tb_journey const& tbj) {
      return tb_to_journey(sched, tbj);
    });
  }

  static void filter_results(std::vector<tb_journey>& results,
                             trip_based_query const& q,
                             trip_based_result const& res) {
    if (!q.is_pretrip()) {
      return;
    }
    utl::erase_if(results, [&](tb_journey const& tbj) {
      return (q.dir_ == search_dir::FWD &&
              tbj.actual_departure_time() > res.interval_end_) ||
             (q.dir_ == search_dir::BWD &&
              tbj.actual_arrival_time() < res.interval_begin_);
    });
  }

  msg_ptr debug(msg_ptr const& msg) const {
    auto const req = motis_content(TripBasedTripDebugRequest, msg);
    auto const& sched = get_schedule();

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_TripBasedTripDebugResponse,
        CreateTripBasedTripDebugResponse(
            fbb,
            fbb.CreateVector(utl::to_vec(*req->trips(),
                                         [&](TripSelectorWrapper const* tsw) {
                                           return get_trip_debug_info(
                                               fbb, *tb_data_, sched, tsw);
                                         })),
            fbb.CreateVector(utl::to_vec(*req->stations(),
                                         [&](flatbuffers::String const* eva) {
                                           return get_station_debug_info(
                                               fbb, *tb_data_, sched,
                                               eva->str());
                                         })))
            .Union());
    return make_msg(fbb);
  }

  std::unique_ptr<tb_data> tb_data_;
};

struct import_state {
  CISTA_COMPARABLE()
  named<cista::hash_t, MOTIS_NAME("schedule_hash")> schedule_hash_;
};

tripbased::tripbased() : module("Trip-Based Routing Options", "tripbased") {
  param(use_data_file_, "use_data_file",
        "create a data_file to speed up subsequent loading");
}

tripbased::~tripbased() = default;

void tripbased::import(motis::module::registry& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "tripbased", reg,
      [this](std::map<std::string, msg_ptr> const& dependencies) {
        using import::ScheduleEvent;
        auto const schedule =
            motis_content(ScheduleEvent, dependencies.at("SCHEDULE"));

        if (!use_data_file_) {
          import_successful_ = true;
          return;
        }

        auto const dir = get_data_directory() / "tripbased";
        boost::filesystem::create_directories(dir);
        auto const filename = dir / "tripbased.bin";

        auto const state = import_state{schedule->hash()};

        auto const& sched = get_schedule();
        update_data_file(sched, filename.generic_string(),
                         read_ini<import_state>(dir / "import.ini") != state);

        import_successful_ = true;
        write_ini(dir / "import.ini", state);
      })
      ->require("SCHEDULE", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

void tripbased::init(motis::module::registry& reg) {
  try {
    if (use_data_file_) {
      auto const filename =
          get_data_directory() / "tripbased" / "tripbased.bin";
      impl_ = std::make_unique<impl>(
          load_data(get_sched(), filename.generic_string()));
    } else {
      impl_ = std::make_unique<impl>(build_data(get_sched()));
    }

    reg.register_op("/tripbased",
                    [this](msg_ptr const& m) { return impl_->route(m); });
    reg.register_op("/tripbased/debug",
                    [this](msg_ptr const& m) { return impl_->debug(m); });

  } catch (std::exception const& e) {
    LOG(logging::warn) << "tripbased module not initialized (" << e.what()
                       << ")";
  }
}

bool tripbased::import_successful() const { return import_successful_; }

tb_data const* tripbased::get_data() const {
  if (impl_) {
    return impl_->tb_data_.get();
  } else {
    return nullptr;
  }
}

}  // namespace motis::tripbased
