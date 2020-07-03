#include "motis/path/prepare/resolve/resolve_sequences.h"

#include <chrono>
#include <iomanip>

#include "boost/algorithm/string/join.hpp"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/resolve/execution_stats.h"
#include "motis/path/prepare/resolve/processing_plan.h"
#include "motis/path/prepare/seq/seq_graph.h"
#include "motis/path/prepare/seq/seq_graph_builder.h"
#include "motis/path/prepare/seq/seq_graph_dijkstra.h"
#include "motis/path/prepare/seq/seq_graph_printer.h"

namespace ml = motis::logging;
namespace sc = std::chrono;

namespace motis::path {

struct result_cache {
  explicit result_cache(routing_result_matrix routing_result)
      : routing_result_{std::move(routing_result)} {}

  routing_result_matrix routing_result_;

  std::mutex mutex_;
  std::map<std::pair<node_ref, node_ref>, std::shared_ptr<osm_path>> path_map_;
};

struct plan_executor {
  plan_executor(processing_plan pp, routing_strategy const* stub_strategy)
      : pp_{std::move(pp)},
        stub_strategy_{stub_strategy},
        open_seq_task_deps_(pp_.seq_tasks_.size()),
        open_part_task_deps_(pp_.part_tasks_.size()),
        result_caches_(pp_.part_tasks_.size()),
        progress_tracker_{utl::get_active_progress_tracker()} {
    for (auto i = 0UL; i < pp_.seq_tasks_.size(); ++i) {
      open_seq_task_deps_[i] = pp_.seq_tasks_[i].part_dependencies_.size();
    }
    for (auto i = 0UL; i < pp_.part_tasks_.size(); ++i) {
      open_part_task_deps_[i] = pp_.part_tasks_[i].seq_dependencies_.size();
    }
  }

  void execute() {
    ml::scoped_timer t{"resolve_sequences"};
    start_ = sc::steady_clock::now();
    progress_tracker_->in_high(pp_.part_task_queue_.size());

    utl::parallel_for_run(
        pp_.part_task_queue_.size(), [&](auto const queue_idx) {
          if (queue_idx % 1000 == 0) {
            dump_queue_stats(queue_idx);
          }
          if (queue_idx % 20000 == 0) {
            stats_.dump_stats();
          }

          auto const part_task_idx = pp_.part_task_queue_[queue_idx];
          auto const& part_task = pp_.part_tasks_[part_task_idx];
          execute_part_task(part_task_idx, part_task);

          for (auto const& seq_task_idx : part_task.seq_dependencies_) {
            if (--open_seq_task_deps_[seq_task_idx] > 0) {
              continue;
            }

            auto const& seq_task = pp_.seq_tasks_[seq_task_idx];
            execute_seq_task(seq_task);

            for (auto const& dep_part_task_idx : seq_task.part_dependencies_) {
              if (--open_part_task_deps_[dep_part_task_idx] > 0) {
                continue;
              }

              auto& cache = result_caches_.at(dep_part_task_idx);
              utl::verify(cache != nullptr, "result_cache does not exist!");
              cache = std::unique_ptr<result_cache>();  // drop
            }
          }
        });

    LOG(ml::info) << "resolving dependencyless sequences (cat: UNKNOWN)";
    for (auto const& seq_task : pp_.seq_tasks_) {
      if (!seq_task.part_dependencies_.empty()) {
        continue;
      }
      execute_seq_task(seq_task);
    }
    stats_.dump_stats();
  }

  void execute_part_task(part_task_idx_t task_idx, part_task const& task) {
    auto start = sc::steady_clock::now();

    auto const& key = task.key_;
    auto const& from_nodes = key.strategy_->close_nodes(key.station_id_from_);
    auto const& to_nodes = key.strategy_->close_nodes(key.station_id_to_);

    auto& part_result = result_caches_.at(task_idx);
    utl::verify(part_result == nullptr, "part_result exists!");

    part_result = std::make_unique<result_cache>(
        key.strategy_->find_routes(from_nodes, to_nodes));

    auto stop = sc::steady_clock::now();
    stats_.add_part_timing(
        {key.strategy_, key.station_id_from_ != key.station_id_to_,
         sc::duration_cast<sc::microseconds>(stop - start).count()});
  }

  void execute_seq_task(seq_task const& task) {
    auto start_sg = sc::steady_clock::now();

    auto seq_graph = build_seq_graph(task);
    // print_seq_graph(pp_.part_tasks_, seq_graph, *task.seq_);

    auto stop_sg = sc::steady_clock::now();

    auto start_sr = sc::steady_clock::now();
    auto const shortest_path = find_shortest_path(seq_graph);
    auto stop_sr = sc::steady_clock::now();

    stats_.add_seq_timing(
        {*task.seq_->classes_.begin(),
         sc::duration_cast<sc::microseconds>(stop_sg - start_sg).count(),
         sc::duration_cast<sc::microseconds>(stop_sr - start_sr).count()});

    mcd::vector<sequence_info> infos;
    mcd::vector<osm_path> paths{task.seq_->station_ids_.size() - 1};
    for (auto const& [edge, new_path] : resolve_sequence(shortest_path)) {
      auto station_idx = edge->from_->station_idx_;
      if (station_idx == paths.size()) {
        --station_idx;  // last inner station belongs to last segment
      }

      auto& path = paths.at(station_idx);
      auto const size_before = path.size();
      path.append(*new_path);

      auto const* s =
          edge->part_task_idx_ == kInvalidPartTask
              ? stub_strategy_
              : pp_.part_tasks_.at(edge->part_task_idx_).key_.strategy_;
      infos.emplace_back(station_idx, size_before, path.size(),
                         edge->from_->station_idx_ != edge->to_->station_idx_,
                         s->source_spec_);
    }

    for (auto& path : paths) {
      path.unique();
      utl::verify(path.size() != 0, "resolve_sequences: empty path");
      path.ensure_line();
    }

    utl::verify(task.seq_->station_ids_.size() == paths.size() + 1,
                "station_ids / paths size mismatch ({} != {})",
                task.seq_->station_ids_.size(), paths.size() + 1);

    auto const lock = std::lock_guard{resolved_seq_mutex_};

    auto cpy = *task.seq_;
    cpy.classes_ = task.classes_;  // maybe subset!
    cpy.paths_ = std::move(paths);
    cpy.sequence_infos_ = std::move(infos);
    resolved_seq_.emplace_back(std::move(cpy));
  }

  seq_graph build_seq_graph(seq_task const& task) {
    seq_graph_builder sg_builder{*task.seq_};

    for (auto const& part_task_idx : task.part_dependencies_) {
      auto const& cache = result_caches_.at(part_task_idx);
      utl::verify(cache != nullptr, "result_cache does not exist! (seq)");

      auto const& key = pp_.part_tasks_[part_task_idx].key_;
      sg_builder.add_part(part_task_idx, key.strategy_, key.station_id_from_,
                          key.station_id_to_, cache->routing_result_);
    }
    sg_builder.add_stub_edges(stub_strategy_);
    return sg_builder.finish();
  }

  std::vector<std::pair<seq_edge const*, std::shared_ptr<osm_path>>>
  resolve_sequence(std::vector<seq_edge const*> const& seq_edges) {
    auto const resolve = [&](auto const* seq_edge, auto maybe_defer) {
      auto const& part_task_idx = seq_edge->part_task_idx_;
      auto const& from_ref = seq_edge->from_->ref_;
      auto const& to_ref = seq_edge->to_->ref_;

      if (part_task_idx == kInvalidPartTask) {
        return std::make_shared<osm_path>(
            stub_strategy_->get_path(from_ref, to_ref));
      }

      auto& cache = result_caches_.at(part_task_idx);
      auto lock = std::unique_lock{cache->mutex_, std::defer_lock};
      if (!maybe_defer) {
        lock.lock();
      } else if (!lock.try_lock()) {
        return std::shared_ptr<osm_path>();
      }

      return utl::get_or_create(
          cache->path_map_, std::make_pair(from_ref, to_ref), [&] {
            auto const* s = pp_.part_tasks_.at(part_task_idx).key_.strategy_;

            auto start_path = sc::steady_clock::now();
            auto path = s->get_path(from_ref, to_ref);
            auto stop_path = sc::steady_clock::now();
            stats_.add_path_timing(
                {s,
                 seq_edge->from_->station_idx_ != seq_edge->to_->station_idx_,
                 sc::duration_cast<sc::microseconds>(stop_path - start_path)
                     .count()});

            return std::make_shared<osm_path>(std::move(path));
          });
    };

    // two phase resolution to prevent mutex contestion
    auto resolved = utl::to_vec(seq_edges, [&](auto const* seq_edge) {
      return std::make_pair(seq_edge, resolve(seq_edge, true));
    });
    for (auto& [seq_edge, path] : resolved) {
      if (!path) {
        path = resolve(seq_edge, false);
      }
    }

    return resolved;
  }

  void dump_queue_stats(size_t curr_part_task) const {
    auto stop = sc::steady_clock::now();
    double t_curr = sc::duration_cast<sc::microseconds>(stop - start_).count();

    auto factor = static_cast<double>(pp_.part_task_queue_.size()) /
                  static_cast<double>(curr_part_task);
    double t_est = t_curr * factor;
    if (curr_part_task == 0) {
      t_est = 0;
    }

    auto const pending_seq_tasks =
        std::count_if(begin(open_seq_task_deps_), end(open_seq_task_deps_),
                      [](auto const& d) { return d > 0; });
    auto const pending_part_tasks =
        std::count_if(begin(open_part_task_deps_), end(open_part_task_deps_),
                      [](auto const& d) { return d > 0; });

    auto const alive_result_caches =
        std::count_if(begin(result_caches_), end(result_caches_),
                      [](auto const& r) { return r != nullptr; });

    progress_tracker_->update(curr_part_task);

    std::clog << "resolve_sequences [" << microsecond_fmt{t_curr} << " | est. "
              << microsecond_fmt{t_est} << "] ("  //
              << std::setw(7) << curr_part_task << "/"  //
              << std::setw(7) << pp_.part_task_queue_.size() << ") "
              << std::setw(7) << pending_seq_tasks << " "  //
              << std::setw(7) << pending_part_tasks << " "  //
              << std::setw(7) << alive_result_caches << "\n";
  }

  struct microsecond_fmt {
    friend std::ostream& operator<<(std::ostream& os,
                                    microsecond_fmt const& o) {
      int t_s = o.t_micro_ / 1e6;

      int t_m = t_s / 60;
      t_s = t_s % 60;

      int t_h = t_m / 60;
      t_m = t_m % 60;

      return os << std::setfill('0') << std::setw(2) << t_h << ":"  //
                << std::setfill('0') << std::setw(2) << t_m << ":"  //
                << std::setfill('0') << std::setw(2) << t_s;
    }
    double t_micro_;
  };

  processing_plan pp_;
  routing_strategy const* stub_strategy_;

  std::vector<std::atomic_size_t> open_seq_task_deps_;
  std::vector<std::atomic_size_t> open_part_task_deps_;

  std::vector<std::unique_ptr<result_cache>> result_caches_;

  std::mutex resolved_seq_mutex_;
  mcd::vector<station_seq> resolved_seq_;

  sc::time_point<sc::steady_clock> start_;
  execution_stats stats_;
  utl::progress_tracker_ptr progress_tracker_;
};

mcd::vector<station_seq> resolve_sequences(
    mcd::vector<station_seq> const& sequences, path_routing& routing) {
  plan_executor executor{make_processing_plan(routing, sequences),
                         routing.get_stub_strategy()};
  executor.execute();
  return std::move(executor.resolved_seq_);
}

}  // namespace motis::path
