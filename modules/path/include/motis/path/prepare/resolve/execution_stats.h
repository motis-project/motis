#pragma once

#include <mutex>
#include <numeric>

#include "fmt/core.h"
#include "fmt/format.h"

#include "utl/equal_ranges_linear.h"

#include "motis/path/prepare/resolve/path_routing.h"

namespace motis::path {

struct strategy_key {
  strategy_key(routing_strategy const* strategy, bool between_stations)
      : strategy_{strategy}, between_stations_{between_stations} {}

  friend bool operator<(strategy_key const& lhs, strategy_key const& rhs) {
    return std::tie(lhs.strategy_, lhs.between_stations_) <
           std::tie(rhs.strategy_, rhs.between_stations_);
  }

  friend bool operator==(strategy_key const& lhs, strategy_key const& rhs) {
    return std::tie(lhs.strategy_, lhs.between_stations_) ==
           std::tie(rhs.strategy_, rhs.between_stations_);
  }

  routing_strategy const* strategy_;
  bool between_stations_;
};

struct strategy_timing {
  strategy_timing(routing_strategy const* strategy, bool between_stations,
                  int64_t mys)
      : key_{strategy, between_stations}, mys_{mys} {}

  strategy_key key_;
  int64_t mys_;
};

struct seq_timing {
  seq_timing(service_class min_clasz, int64_t graph_mys, int64_t route_mys)
      : min_clasz_{min_clasz}, graph_mys_{graph_mys}, route_mys_{route_mys} {}

  service_class min_clasz_;
  int64_t graph_mys_;
  int64_t route_mys_;
};

struct execution_stats {
  void add_part_timing(strategy_timing t) {
    auto const lock = std::lock_guard{buffer_mutex_};
    part_timings_buffer_.push_back(t);
  }

  void add_path_timing(strategy_timing t) {
    auto const lock = std::lock_guard{buffer_mutex_};
    path_timings_buffer_.push_back(t);
  }

  void add_seq_timing(seq_timing t) {
    auto const lock = std::lock_guard{buffer_mutex_};
    seq_timings_buffer_.push_back(t);
  }

  void dump_stats() {
    auto const lock = std::lock_guard{main_mutex_};
    read_buffers();

    fmt::memory_buffer out;

    format_headline(out);
    for (auto& [key, v] : main_part_timings_) {
      fmt::format_to(out, "[PART] {:>14} {} ",
                     key.strategy_->source_spec_.str(),
                     key.between_stations_ ? 'B' : 'W');
      format_timing_summary(out, v);
    }
    format_headline(out);
    for (auto& [min_cat, v] : seq_graph_timings_) {
      fmt::format_to(out, "[SEQ|GRAPH] cat {:>7} ", min_cat);
      format_timing_summary(out, v);
    }
    format_headline(out);
    for (auto& [min_cat, v] : seq_route_timings_) {
      fmt::format_to(out, "[SEQ|ROUTE] cat {:>7} ", min_cat);
      format_timing_summary(out, v);
    }
    format_headline(out);
    for (auto& [key, v] : main_path_timings_) {
      fmt::format_to(out, "[PATH] {:>14} {} ",
                     key.strategy_->source_spec_.str(),
                     key.between_stations_ ? 'B' : 'W');
      format_timing_summary(out, v);
    }

    std::clog << fmt::to_string(out) << std::endl;
  }

  static void format_headline(fmt::memory_buffer& out) {
    fmt::format_to(out, "{:23} | {:8} | {:9} | {:9} | {:9} | {:9} | {:9}\n",
                   "task", "count", "sum", "avg", "sd", "q50", "q95");
  }

  static void format_timing_summary(fmt::memory_buffer& out,
                                    std::vector<double>& v) {
    auto const format_count = [&](double const n) {
      auto const k = n / 1e3;
      auto const m = n / 1e6;
      auto const g = n / 1e9;
      if (n < 1e3) {
        fmt::format_to(out, "| {:>7}  ", n);
      } else if (k < 1e3) {
        fmt::format_to(out, "| {:>7.1f}K ", k);
      } else if (m < 1e3) {
        fmt::format_to(out, "| {:>7.1f}M ", m);
      } else {
        fmt::format_to(out, "| {:>7.1f}G ", g);
      }
    };

    auto const format_dur = [&](double const mys) {
      auto const ms = mys / 1e3;
      auto const s = mys / 1e6;
      if (mys < 1e3) {
        fmt::format_to(out, "| {:>7.3f}µs ", mys);
      } else if (ms < 1e3) {
        fmt::format_to(out, "| {:>7.3f}ms ", ms);
      } else if (s < 1e3) {
        fmt::format_to(out, "| {:>7.3f}s  ", s);
      } else if (s < 1e4) {
        fmt::format_to(out, "| {:>7.2f}s  ", s);
      } else if (s < 1e5) {
        fmt::format_to(out, "| {:>7.1f}s  ", s);
      } else {
        fmt::format_to(out, "| {:>7.0f}s  ", s);
      }
    };

    std::sort(begin(v), end(v));

    auto const count = v.size();
    double const sum = std::accumulate(begin(v), end(v), 0.);
    double const mean = sum / count;
    double const tmp = std::accumulate(
        begin(v), end(v), 0.,
        [&](auto a, auto val) { return a + (val - mean) * (val - mean); });
    double const sd = std::sqrt(tmp / count);

    auto const q50 = v[v.size() * .5];
    auto const q95 = v[v.size() * .95];

    format_count(count);
    format_dur(sum);
    format_dur(mean);
    format_dur(sd);
    format_dur(q50);
    format_dur(q95);
    fmt::format_to(out, "\n");
  }

private:
  void read_buffers() {
    auto const lock = std::lock_guard{buffer_mutex_};
    auto const read_strategy_buffer = [&](auto& buf, auto& map) {
      std::sort(begin(buf), end(buf),
                [](auto const& a, auto const& b) { return a.key_ < b.key_; });
      utl::equal_ranges_linear(
          buf, [](auto const& a, auto const& b) { return a.key_ == b.key_; },
          [&](auto lb, auto ub) {
            auto& vec = map[lb->key_];
            for (auto it = lb; it != ub; ++it) {
              vec.push_back(it->mys_);
            }
            // XXX maybe use std::inplace_merge instead of later sort
          });

      buf.clear();
    };

    read_strategy_buffer(part_timings_buffer_, main_part_timings_);
    read_strategy_buffer(path_timings_buffer_, main_path_timings_);

    std::sort(begin(seq_timings_buffer_), end(seq_timings_buffer_),
              [](auto const& a, auto const& b) {
                return a.min_clasz_ < b.min_clasz_;
              });
    utl::equal_ranges_linear(
        seq_timings_buffer_,
        [](auto const& a, auto const& b) {
          return a.min_clasz_ == b.min_clasz_;
        },
        [&](auto lb, auto ub) {
          auto& graph_vec =
              seq_graph_timings_[static_cast<service_class_t>(lb->min_clasz_)];
          auto& route_vec =
              seq_route_timings_[static_cast<service_class_t>(lb->min_clasz_)];
          for (auto it = lb; it != ub; ++it) {
            graph_vec.push_back(it->graph_mys_);
            route_vec.push_back(it->route_mys_);
          }
          // XXX maybe use std::inplace_merge instead of later sort
        });

    seq_timings_buffer_.clear();
  }

  std::mutex main_mutex_;
  std::map<strategy_key, std::vector<double>> main_part_timings_;
  std::map<strategy_key, std::vector<double>> main_path_timings_;

  std::map<int, std::vector<double>> seq_graph_timings_;
  std::map<int, std::vector<double>> seq_route_timings_;

  std::mutex buffer_mutex_;
  std::vector<strategy_timing> part_timings_buffer_;
  std::vector<strategy_timing> path_timings_buffer_;
  std::vector<seq_timing> seq_timings_buffer_;
};

}  // namespace motis::path
