#include <chrono>
#include <algorithm>
#include <atomic>
#include <locale>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "utl/progress_tracker.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/tripbased/data.h"
#include "motis/tripbased/preprocessing.h"
#include "motis/tripbased/serialization.h"

using namespace motis::access;
using namespace motis::logging;

namespace fs = boost::filesystem;

namespace motis::tripbased {

struct thousands_sep : std::numpunct<char> {
  char_type do_thousands_sep() const override { return ','; }
  string_type do_grouping() const override { return "\3"; }
};

edge const* get_outgoing_route_edge(node const* route_node) {
  assert(route_node != nullptr);
  assert(route_node->is_route_node());
  for (auto const& e : route_node->edges_) {
    if (e.type() == edge::ROUTE_EDGE) {
      return &e;
    }
  }
  return nullptr;
}

edge const* get_incoming_route_edge(node const* route_node) {
  assert(route_node != nullptr);
  assert(route_node->is_route_node());
  for (auto const& e : route_node->incoming_edges_) {
    if (e->type() == edge::ROUTE_EDGE) {
      return e;
    }
  }
  return nullptr;
}

bool is_in_allowed(node const* route_node) {
  for (auto const& e : route_node->incoming_edges_) {
    if (e->from_->is_station_node() && e->type() != edge::INVALID_EDGE) {
      return true;
    }
  }
  return false;
}

bool is_out_allowed(node const* route_node) {
  for (auto const& e : route_node->edges_) {
    if (e.to_->is_station_node() && e.type() != edge::INVALID_EDGE) {
      return true;
    }
  }
  return false;
}

struct preprocessing {
  preprocessing(schedule const& sched, tb_data& data)
      : sched_(sched),
        data_(data),
        progress_tracker_{
            utl::get_active_progress_tracker_or_activate("tripbased")} {}

  void init() {
    scoped_timer timer{"trip-based preprocessing: init"};
    auto const prev_locale =
        std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));
    auto const stop_count = sched_.station_nodes_.size();
    auto const line_count = sched_.expanded_trips_.index_size() - 1;
    data_.line_to_first_trip_.reserve(line_count);
    data_.line_to_last_trip_.reserve(line_count);
    data_.line_stop_count_.reserve(line_count);
    data_.footpaths_.reserve_index(stop_count);
    data_.reverse_footpaths_.reserve_index(stop_count);
    data_.lines_at_stop_.reserve_index(stop_count);
    data_.stops_on_line_.reserve_index(stop_count);
    data_.in_allowed_.reserve_index(line_count);
    auto trip_idx = 0UL;
    auto lcon_count = 0UL;
    LOG(info) << "trip-based preprocessing:";
    LOG(info) << stop_count << " stops";
    LOG(info) << sched_.expanded_trips_.data_size() << " motis trips";
    LOG(info) << line_count << " lines";

    std::vector<std::vector<std::pair<line_id, stop_idx_t>>> lines_at_stop;
    lines_at_stop.resize(stop_count);

    progress_tracker_->status("Init: Routes");
    for (auto route_idx = 0UL; route_idx < line_count; ++route_idx) {
      auto const& route_trips = sched_.expanded_trips_[route_idx];
      utl::verify(!route_trips.empty(), "empty route");
      auto const first_trip = route_trips[0];
      auto const first_trip_id = route_trips.data_index(0);

      utl::verify(data_.line_to_first_trip_.size() == route_idx,
                  "line to first trip index invalid");
      utl::verify(data_.line_to_last_trip_.size() == route_idx,
                  "line to last trip index invalid");
      data_.line_to_first_trip_.push_back(first_trip_id);
      trip_idx = first_trip_id;
      for (auto i = 0UL; i < route_trips.size(); ++i) {
        trip_idx = route_trips.data_index(i);
        utl::verify(data_.trip_to_line_.size() == trip_idx,
                    "incorrect trip index in trip to line");
        data_.trip_to_line_.push_back(route_idx);
      }

      data_.line_to_last_trip_.push_back(trip_idx);
      ++trip_idx;

      stop_idx_t line_stop_count = 0U;
      for (auto const& stop : stops(first_trip)) {
        auto const* rn = stop.get_route_node();
        const auto station_id = stop.get_station_id();
        data_.stops_on_line_.push_back(station_id);
        data_.in_allowed_.push_back(
            static_cast<uint8_t>(is_in_allowed(rn) ? 1U : 0U));
        data_.out_allowed_.push_back(
            static_cast<uint8_t>(is_out_allowed(rn) ? 1U : 0U));
        lines_at_stop[station_id].emplace_back(route_idx, line_stop_count);
        ++line_stop_count;
      }
      utl::verify(data_.stops_on_line_.current_key() == route_idx,
                  "incorrect line index in stops on line");
      utl::verify(line_stop_count == first_trip->edges_->size() + 1,
                  "invalid line stop count");
      data_.stops_on_line_.finish_key();
      data_.in_allowed_.finish_key();
      data_.out_allowed_.finish_key();
      data_.line_stop_count_.push_back(line_stop_count);

      for (auto i = 0UL; i < route_trips.size(); ++i) {
        data_.arrival_times_.push_back(INVALID_TIME);
        auto last_time = 0U;
        for (auto const& sec : sections(route_trips[i])) {
          auto const& lc = sec.lcon();
          utl::verify(lc.d_time_ <= lc.a_time_,
                      "route has invalid timestamps (1)");
          utl::verify(last_time <= lc.d_time_,
                      "route has invalid timestamps (2)");
          data_.arrival_times_.push_back(lc.a_time_);
          data_.departure_times_.push_back(lc.d_time_);
          last_time = lc.a_time_;
          ++lcon_count;
        }
        data_.departure_times_.push_back(INVALID_TIME);
        utl::verify(data_.arrival_times_.current_key() == first_trip_id + i,
                    "incorrect arrival times index");
        utl::verify(data_.departure_times_.current_key() == first_trip_id + i,
                    "incorrect departure times index");
        data_.arrival_times_.finish_key();
        data_.departure_times_.finish_key();
      }
    }

    progress_tracker_->status("Init: Lines at Stop");
    for (auto const& lines : lines_at_stop) {
      for (auto const& [line, stop_idx] : lines) {
        utl::verify(stop_idx < data_.line_stop_count_[line],
                    "lines at stop stop index bounds check");
        data_.lines_at_stop_.emplace_back(line, stop_idx);
      }
      data_.lines_at_stop_.finish_key();
    }

    progress_tracker_->status("Init: Station Nodes");
    for (auto const& st : sched_.station_nodes_) {
      for (auto const& fp : sched_.stations_[st->id_]->outgoing_footpaths_) {
        data_.footpaths_.emplace_back(fp);
      }
      for (auto const& fp : sched_.stations_[st->id_]->incoming_footpaths_) {
        data_.reverse_footpaths_.emplace_back(fp);
      }
      utl::verify(data_.footpaths_.current_key() == st->id_,
                  "incorrect footpath index");
      data_.footpaths_.finish_key();
      utl::verify(data_.reverse_footpaths_.current_key() == st->id_,
                  "incorrect reverse footpath index");
      data_.reverse_footpaths_.finish_key();
    }

    // add sentinel values
    data_.footpaths_.finish_map();
    data_.reverse_footpaths_.finish_map();
    data_.lines_at_stop_.finish_map();
    data_.stops_on_line_.finish_map();
    data_.arrival_times_.finish_map();
    data_.departure_times_.finish_map();
    data_.in_allowed_.finish_map();
    data_.out_allowed_.finish_map();

    data_.trip_count_ = trip_idx;
    data_.line_count_ = line_count;

    LOG(info) << lcon_count << " light connections";
    LOG(info) << data_.footpaths_.data_size() << " footpaths";

    utl::verify(data_.trip_count_ == sched_.expanded_trips_.data_size(),
                "incorrect trip count");
    utl::verify(data_.line_to_first_trip_.size() == line_count,
                "incorrect size of line to first trip");
    utl::verify(data_.line_stop_count_.size() == line_count,
                "incorrect size of line stop count");
    utl::verify(data_.trip_to_line_.size() == data_.trip_count_,
                "incorrect size of trip to line");
    utl::verify(
        data_.footpaths_.index_size() == data_.reverse_footpaths_.index_size(),
        "different number of footpaths and reverse footpaths");
    utl::verify(data_.footpaths_.index_size() == stop_count + 1,
                "incorrect size of footpath index");
    utl::verify(data_.lines_at_stop_.index_size() == stop_count + 1,
                "incorrect size of lines at stop");
    utl::verify(data_.stops_on_line_.index_size() == line_count + 1,
                "incorrect size of stops on line");
    utl::verify(data_.arrival_times_.index_size() == data_.trip_count_ + 1,
                "incorrect size of arrival times");
    utl::verify(data_.departure_times_.index_size() == data_.trip_count_ + 1,
                "incorrect size of departure times");
    utl::verify(data_.in_allowed_.index_size() == line_count + 1,
                "incorrect size of in allowed");
    utl::verify(data_.out_allowed_.index_size() == line_count + 1,
                "incorrect size of out allowed");
    utl::verify(data_.footpaths_.finished(), "footpaths not finished");
    utl::verify(data_.reverse_footpaths_.finished(),
                "reverse not footpaths finished");
    utl::verify(data_.lines_at_stop_.finished(), "lines at stop not finished");
    utl::verify(data_.stops_on_line_.finished(), "stops on line not finished");
    utl::verify(data_.arrival_times_.finished(), "arrival times not finished");
    utl::verify(data_.in_allowed_.finished(), "in allowed not finished");
    std::cout.imbue(prev_locale);
  }

  void precompute() {
    precompute_transfers();
    precompute_reverse_transfers();
  }

  void precompute_transfers() {
    progress_tracker_->status("Transfers: FWD").out_bounds(0.F, 50.F);

    scoped_timer timer{"trip-based preprocessing: precompute transfers"};
    auto const prev_locale =
        std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));
    LOG(info) << "precompute transfers: " << data_.trip_count_ << " trips, "
              << data_.line_count_ << " lines";

    last_progress_update_ =
        std::chrono::steady_clock::now() - std::chrono::minutes(1);

    auto const thread_count = std::thread::hardware_concurrency();
    if (data_.trip_count_ > thread_count) {
      std::vector<std::thread> threads;
      threads.reserve(thread_count);
      for (auto t = 0U; t < thread_count; ++t) {
        threads.emplace_back([=]() {
          precompute_transfers_thread(static_cast<trip_id>(t),
                                      static_cast<trip_id>(thread_count));
        });
      }
      for (auto& t : threads) {
        t.join();
      }
      assert(transfers_queue_.empty());
    } else {
      precompute_transfers_thread(0, 1);
    }
    data_.transfers_.finish_map();

    LOG(info) << data_.transfers_.data_size() << " transfers - "
              << (uturns_ + no_improvements_) << " ignored (" << uturns_
              << " u-turns + " << no_improvements_ << " no improvements)";
    assert(data_.transfers_.finished());
    std::cout.imbue(prev_locale);
  }

  void precompute_reverse_transfers() {
    progress_tracker_->status("Transfers: BWD").out_bounds(50.F, 100.F);

    scoped_timer timer{
        "trip-based preprocessing: precompute reverse transfers"};
    expected_trip_id_ = 0;
    uturns_ = 0;
    no_improvements_ = 0;
    auto const prev_locale =
        std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));
    LOG(info) << "precompute reverse transfers: " << data_.trip_count_
              << " trips, " << data_.line_count_ << " lines";

    last_progress_update_ =
        std::chrono::steady_clock::now() - std::chrono::minutes(1);

    data_.reverse_transfers_.reserve_data(data_.transfers_.data_size());

    auto const thread_count = std::thread::hardware_concurrency();
    if (data_.trip_count_ > thread_count) {
      std::vector<std::thread> threads;
      threads.reserve(thread_count);
      for (auto t = 0U; t < thread_count; ++t) {
        threads.emplace_back([=]() {
          precompute_reverse_transfers_thread(
              static_cast<trip_id>(t), static_cast<trip_id>(thread_count));
        });
      }
      for (auto& t : threads) {
        t.join();
      }
      assert(transfers_queue_.empty());
    } else {
      precompute_reverse_transfers_thread(0, 1);
    }
    data_.reverse_transfers_.finish_map();

    LOG(info) << data_.reverse_transfers_.data_size() << " reverse transfers - "
              << (uturns_ + no_improvements_) << " ignored (" << uturns_
              << " u-turns + " << no_improvements_ << " no improvements)";
    assert(data_.reverse_transfers_.finished());
    std::cout.imbue(prev_locale);
  }

private:
  void precompute_transfers_thread(trip_id first_trip_idx, trip_id stride) {
    auto const stop_count = sched_.stations_.size();
    std::vector<time> earliest_arrival(stop_count);
    std::vector<time> earliest_change(stop_count);

    for (uint64_t trip_idx = first_trip_idx; trip_idx < data_.trip_count_;
         trip_idx += stride) {
      auto const line_idx = data_.trip_to_line_[trip_idx];
      auto const out_allowed = data_.out_allowed_[line_idx];

      auto const line_stop_count = data_.line_stop_count_[line_idx];
      auto const line_stops = data_.stops_on_line_[line_idx];

      std::vector<std::vector<tb_transfer>> transfers(line_stop_count);

      std::fill(begin(earliest_arrival), end(earliest_arrival), INVALID_TIME);
      std::fill(begin(earliest_change), end(earliest_change), INVALID_TIME);

      for (auto from_stop_idx = line_stop_count - 1; from_stop_idx > 0;
           --from_stop_idx) {
        auto const station_idx = line_stops[from_stop_idx];

        auto const trip_arrival = data_.arrival_times_[trip_idx][from_stop_idx];
        if (out_allowed[from_stop_idx] == 0) {
          continue;
        }

        if (trip_arrival < earliest_arrival[station_idx]) {
          earliest_arrival[station_idx] = trip_arrival;
        }

        auto const footpaths = outgoing_footpaths(station_idx);

        for (auto const& fp : footpaths) {
          auto const fp_arrival =
              static_cast<time>(trip_arrival + fp.duration_);
          if (fp_arrival < earliest_arrival[fp.to_stop_]) {
            earliest_arrival[fp.to_stop_] = fp_arrival;
          }
          if (fp_arrival < earliest_change[fp.to_stop_]) {
            earliest_change[fp.to_stop_] = fp_arrival;
          }
        }

        for (auto const& fp : footpaths) {
          auto const station_arrival =
              static_cast<time>(trip_arrival + fp.duration_);
          for (auto const& [other_line, other_stop_idx, _] :
               data_.lines_at_stop_[fp.to_stop_]) {
            (void)_;
            if (is_last_stop_of_line(other_line, other_stop_idx) ||
                data_.in_allowed_[other_line][other_stop_idx] == 0) {
              continue;
            }
            auto const reachable = data_.first_reachable_trip(
                other_line, other_stop_idx, station_arrival);
            if (!reachable || (reachable->second - trip_arrival) > 1440) {
              continue;
            }
            auto const other_trip = reachable->first;
            if (other_line != line_idx || other_stop_idx < from_stop_idx ||
                other_trip < trip_idx) {
              // don't add u-turn transfers
              assert(from_stop_idx > 0);
              utl::verify(other_stop_idx < data_.line_stop_count_[other_line],
                          "invalid other stop index 1");
              auto const from_prev_stop =
                  data_.stops_on_line_[line_idx][from_stop_idx - 1];
              auto const to_next_stop =
                  data_.stops_on_line_[other_line][other_stop_idx + 1];
              if (from_prev_stop == to_next_stop &&
                  out_allowed[from_stop_idx - 1] != 0 &&
                  data_.in_allowed_[other_line][other_stop_idx + 1] != 0 &&
                  (data_.arrival_times_[trip_idx][from_stop_idx - 1] +
                       sched_.stations_[from_prev_stop]->transfer_time_ <=
                   data_.departure_times_[other_trip][other_stop_idx + 1])) {
                ++uturns_;
                continue;
              }
              if (!keep_transfer(other_line, other_trip, other_stop_idx,
                                 earliest_arrival, earliest_change)) {
                ++no_improvements_;
                continue;
              }
              transfers[from_stop_idx].emplace_back(other_trip, other_stop_idx);
            }
          }
        }
      }

      add_transfers(trip_idx, std::move(transfers));
    }
  }

  void precompute_reverse_transfers_thread(trip_id first_trip_idx,
                                           trip_id stride) {
    auto const stop_count = sched_.stations_.size();
    std::vector<time> latest_departure(stop_count);
    std::vector<time> latest_change(stop_count);

    for (uint64_t trip_idx = first_trip_idx; trip_idx < data_.trip_count_;
         trip_idx += stride) {
      auto const line_idx = data_.trip_to_line_[trip_idx];
      auto const in_allowed = data_.in_allowed_[line_idx];

      auto const line_stop_count = data_.line_stop_count_[line_idx];
      auto const line_stops = data_.stops_on_line_[line_idx];

      std::vector<std::vector<tb_reverse_transfer>> transfers(line_stop_count);

      std::fill(begin(latest_departure), end(latest_departure), 0);
      std::fill(begin(latest_change), end(latest_change), 0);

      for (int to_stop_idx = 0; to_stop_idx <= line_stop_count - 2;
           ++to_stop_idx) {
        auto const station_idx = line_stops[to_stop_idx];

        auto const trip_departure =
            data_.departure_times_[trip_idx][to_stop_idx];
        if (in_allowed[to_stop_idx] == 0) {
          continue;
        }

        if (trip_departure > latest_departure[station_idx]) {
          latest_departure[station_idx] = trip_departure;
        }

        auto const footpaths = incoming_footpaths(station_idx);

        for (auto const& fp : footpaths) {
          auto const fp_departure =
              static_cast<time>(trip_departure - fp.duration_);
          if (fp_departure > latest_departure[fp.from_stop_]) {
            latest_departure[fp.from_stop_] = fp_departure;
          }
          if (fp_departure > latest_change[fp.from_stop_]) {
            latest_change[fp.from_stop_] = fp_departure;
          }
        }

        for (auto const& fp : footpaths) {
          auto const station_departure =
              static_cast<time>(trip_departure - fp.duration_);
          for (auto const& [other_line, other_stop_idx, _] :
               data_.lines_at_stop_[fp.from_stop_]) {
            (void)_;
            if (other_stop_idx == 0 ||
                data_.out_allowed_[other_line][other_stop_idx] == 0) {
              continue;
            }
            auto const reachable = data_.last_reachable_trip(
                other_line, other_stop_idx, station_departure);
            if (!reachable || (trip_departure - reachable->second) > 1440) {
              continue;
            }
            auto const other_trip = reachable->first;
            if (other_line != line_idx || to_stop_idx < other_stop_idx ||
                trip_idx < other_trip) {
              // don't add u-turn transfers
              utl::verify(other_stop_idx > 0, "invalid other stop index 2");
              utl::verify(other_stop_idx < data_.line_stop_count_[other_line],
                          "invalid other stop index 3");
              auto const to_next_stop =
                  data_.stops_on_line_[line_idx][to_stop_idx + 1];
              auto const from_prev_stop =
                  data_.stops_on_line_[other_line][other_stop_idx - 1];
              if (from_prev_stop == to_next_stop &&
                  in_allowed[to_stop_idx + 1] != 0 &&
                  data_.out_allowed_[other_line][other_stop_idx - 1] != 0 &&
                  (data_.departure_times_[trip_idx][to_stop_idx + 1] -
                       sched_.stations_[to_next_stop]->transfer_time_ >=
                   data_.arrival_times_[other_trip][other_stop_idx - 1])) {
                ++uturns_;
                continue;
              }
              if (!keep_reverse_transfer(other_line, other_trip, other_stop_idx,
                                         latest_departure, latest_change)) {
                ++no_improvements_;
                continue;
              }
              transfers[to_stop_idx].emplace_back(other_trip, other_stop_idx,
                                                  to_stop_idx);
            }
          }
        }
      }

      assert(transfers.size() == line_stop_count);
      add_reverse_transfers(trip_idx, std::move(transfers));
    }
  }

  void add_transfers(trip_id trip_idx,
                     std::vector<std::vector<tb_transfer>> trip_transfers) {
    std::lock_guard<std::mutex> guard{transfers_mutex_};

    auto const add_to_transfers =
        [&](trip_id trip,
            std::vector<std::vector<tb_transfer>> const& transfers) {
          for (auto const& stop_transfers : transfers) {
            for (auto const& t : stop_transfers) {
              data_.transfers_.push_back(t);
            }
            data_.transfers_.finish_nested_key();
          }
          data_.transfers_.finish_base_key();
          update_progress(trip, data_.transfers_.data_size());
        };

    if (trip_idx == expected_trip_id_) {
      add_to_transfers(trip_idx, trip_transfers);
      ++expected_trip_id_;
      auto it = transfers_queue_.find(expected_trip_id_);
      while (it != end(transfers_queue_)) {
        add_to_transfers(it->first, it->second);
        transfers_queue_.erase(it);
        ++expected_trip_id_;
        it = transfers_queue_.find(expected_trip_id_);
      }
    } else {
      transfers_queue_[trip_idx] = std::move(trip_transfers);
    }
  }

  void add_reverse_transfers(
      trip_id trip_idx,
      std::vector<std::vector<tb_reverse_transfer>> trip_transfers) {
    std::lock_guard<std::mutex> guard{transfers_mutex_};

    auto const add_to_transfers =
        [&](trip_id trip,
            std::vector<std::vector<tb_reverse_transfer>> const& transfers) {
          for (auto const& stop_transfers : transfers) {
            for (auto const& t : stop_transfers) {
              data_.reverse_transfers_.push_back(t);
            }
            data_.reverse_transfers_.finish_nested_key();
          }
          data_.reverse_transfers_.finish_base_key();

          update_progress(trip, data_.reverse_transfers_.data_size());
        };

    if (trip_idx == expected_trip_id_) {
      add_to_transfers(trip_idx, trip_transfers);
      ++expected_trip_id_;
      auto it = reverse_transfers_queue_.find(expected_trip_id_);
      while (it != end(reverse_transfers_queue_)) {
        add_to_transfers(it->first, it->second);
        reverse_transfers_queue_.erase(it);
        ++expected_trip_id_;
        it = reverse_transfers_queue_.find(expected_trip_id_);
      }
    } else {
      reverse_transfers_queue_[trip_idx] = std::move(trip_transfers);
    }
  }

  void update_progress(trip_id trip, std::size_t transfer_count) {
    auto const now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_progress_update_)
            .count() >= 1000) {
      auto const percentage =
          static_cast<int>(std::round(100.0 * static_cast<double>(trip + 1) /
                                      static_cast<double>(data_.trip_count_)));
      progress_tracker_->update(percentage);
      LOG(info) << percentage << "% - " << (trip + 1) << "/"
                << data_.trip_count_ << " trips... " << transfer_count
                << " transfers, " << (uturns_ + no_improvements_)
                << " ignored (" << uturns_ << " u-turns + " << no_improvements_
                << " no improvements)";
      last_progress_update_ = now;
    }
  }

  bool keep_transfer(line_id to_line, trip_id to_trip, stop_idx_t enter_index,
                     std::vector<time>& earliest_arrival,
                     std::vector<time>& earliest_change) {
    auto const to_stop_count = data_.line_stop_count_[to_line];
    auto const arrival_times = data_.arrival_times_[to_trip];
    auto const stops_on_line = data_.stops_on_line_[to_line];
    auto const out_allowed = data_.out_allowed_[to_line];
    bool keep = false;
    for (auto stop_idx = enter_index + 1; stop_idx < to_stop_count;
         ++stop_idx) {
      auto const trip_arrival = arrival_times[stop_idx];
      auto const station = stops_on_line[stop_idx];
      if (out_allowed[stop_idx] == 0) {
        continue;
      }
      if (trip_arrival < earliest_arrival[station]) {
        earliest_arrival[station] = trip_arrival;
        keep = true;
      }
      for (auto const& fp : outgoing_footpaths(station)) {
        auto const fp_arrival = static_cast<time>(trip_arrival + fp.duration_);
        if (fp_arrival < earliest_arrival[fp.to_stop_]) {
          earliest_arrival[fp.to_stop_] = fp_arrival;
          keep = true;
        }
        if (fp_arrival < earliest_change[fp.to_stop_]) {
          earliest_change[fp.to_stop_] = fp_arrival;
          keep = true;
        }
      }
    }
    return keep;
  }

  bool keep_reverse_transfer(line_id from_line, trip_id from_trip,
                             stop_idx_t exit_index,
                             std::vector<time>& latest_departure,
                             std::vector<time>& latest_change) {
    auto const departure_times = data_.departure_times_[from_trip];
    auto const stops_on_line = data_.stops_on_line_[from_line];
    auto const in_allowed = data_.in_allowed_[from_line];
    bool keep = false;
    for (int stop_idx = exit_index - 1; stop_idx >= 0; --stop_idx) {
      auto const trip_departure = departure_times[stop_idx];
      auto const station = stops_on_line[stop_idx];
      if (in_allowed[stop_idx] == 0) {
        continue;
      }
      if (trip_departure > latest_departure[station]) {
        latest_departure[station] = trip_departure;
        keep = true;
      }
      for (auto const& fp : incoming_footpaths(station)) {
        auto const fp_departure =
            static_cast<time>(trip_departure - fp.duration_);
        if (fp_departure > latest_departure[fp.from_stop_]) {
          latest_departure[fp.from_stop_] = fp_departure;
          keep = true;
        }
        if (fp_departure > latest_change[fp.from_stop_]) {
          latest_change[fp.from_stop_] = fp_departure;
          keep = true;
        }
      }
    }
    return keep;
  }

  std::vector<tb_footpath> outgoing_footpaths(station_id from_stop_idx) {
    std::vector<tb_footpath> reachable;
    auto const& footpaths = data_.footpaths_[from_stop_idx];
    reachable.reserve(1 + footpaths.size());
    reachable.emplace_back(from_stop_idx, from_stop_idx,
                           sched_.stations_[from_stop_idx]->transfer_time_);
    std::copy(begin(footpaths), end(footpaths), std::back_inserter(reachable));
    return reachable;
  }

  std::vector<tb_footpath> incoming_footpaths(station_id to_stop_idx) {
    std::vector<tb_footpath> reachable;
    auto const& footpaths = data_.reverse_footpaths_[to_stop_idx];
    reachable.reserve(1 + footpaths.size());
    reachable.emplace_back(to_stop_idx, to_stop_idx,
                           sched_.stations_[to_stop_idx]->transfer_time_);
    std::copy(begin(footpaths), end(footpaths), std::back_inserter(reachable));
    return reachable;
  }

  bool is_last_stop_of_line(line_id line, stop_idx_t stop_idx) {
    assert(line < data_.line_count_);
    auto const stop_count = data_.line_stop_count_[line];
    assert(stop_idx < stop_count);
    return stop_idx == stop_count - 1;
  }

  schedule const& sched_;
  tb_data& data_;
  utl::progress_tracker_ptr progress_tracker_;
  std::atomic<uint64_t> uturns_{0};
  std::atomic<uint64_t> no_improvements_{0};
  std::mutex transfers_mutex_;
  trip_id expected_trip_id_{0};
  std::map<trip_id, std::vector<std::vector<tb_transfer>>> transfers_queue_;
  std::map<trip_id, std::vector<std::vector<tb_reverse_transfer>>>
      reverse_transfers_queue_;
  std::vector<std::vector<tb_footpath>> outgoing_footpaths_;
  std::vector<std::vector<tb_footpath>> incoming_footpaths_;
  std::chrono::time_point<std::chrono::steady_clock> last_progress_update_;
};

std::unique_ptr<tb_data> build_data(schedule const& sched) {
  auto data = std::make_unique<tb_data>();
  preprocessing pp(sched, *data);
  pp.init();
  pp.precompute();
  LOG(info) << "trip-based preprocessing complete:";
  LOG(info) << data->line_count_ << " lines";
  LOG(info) << data->trip_count_ << " trips";
  LOG(info) << data->transfers_.data_size() << " transfers";
  LOG(info) << data->reverse_transfers_.data_size() << " reverse transfers";
  return data;
}

std::unique_ptr<tb_data> load_data(schedule const& sched,
                                   std::string const& filename) {
  utl::verify(!filename.empty(), "update_data_file: filename empty");
  utl::verify(fs::exists(filename), "update_data_file: file does not exist {}",
              filename);
  return serialization::read_data(filename, sched);
}

void update_data_file(schedule const& sched, std::string const& filename,
                      bool const force_update) {
  utl::verify(!filename.empty(), "update_data_file: filename empty");

  if (!force_update && fs::exists(filename)) {
    LOG(info) << "loading trip-based data from file " << filename;
    scoped_timer load_timer{"trip-based deserialization"};
    if (serialization::data_okay_for_schedule(filename, sched)) {
      return;
    } else {
      LOG(info) << "existing trip-based data is not okay: " << filename;
    }
  }

  LOG(info) << "calculating trip-based data...";
  auto data = build_data(sched);
  LOG(info) << "writing trip-based data to file " << filename;
  scoped_timer write_timer{"trip-based serialization"};
  serialization::write_data(*data, filename, sched);
}

}  // namespace motis::tripbased
