#include "motis/loader/footpaths.h"

#include <stack>

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/parallel_for.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

namespace ml = motis::logging;

namespace motis::loader {

constexpr auto kNoComponent = std::numeric_limits<uint32_t>::max();

// station_idx -> [footpath, ...]
using footgraph = std::vector<std::vector<footpath>>;

// (component, original station_idx)
using component_vec = std::vector<std::pair<uint32_t, uint32_t>>;
using component_it = component_vec::iterator;
using component_range = std::pair<component_it, component_it>;

std::string to_str(std::vector<footpath> const& footpaths) {
  std::stringstream s;
  for (auto const& [i, fp] : utl::enumerate(footpaths)) {
    s << (i == 0 ? "" : ", ") << fp.from_station_ << "-" << fp.to_station_
      << "-" << fp.duration_ << "min";
  }
  return s.str();
}

footgraph get_footpath_graph(
    mcd::vector<station_node_ptr> const& station_nodes) {
  return utl::to_vec(station_nodes, [](auto const& station_node) {
    std::vector<footpath> fps;
    if (station_node->foot_node_ != nullptr) {
      for (auto const& foot_edge : station_node->foot_node_->edges_) {
        auto const from_station = foot_edge.from_->get_station()->id_;
        auto const to_station = foot_edge.to_->get_station()->id_;
        auto const duration = foot_edge.m_.foot_edge_.time_cost_;

        utl::verify(from_station == station_node->id_,
                    "foot path wrong at station");
        if (from_station != to_station) {
          fps.emplace_back(footpath{from_station, to_station, duration});
        }
      }

      std::sort(begin(fps), end(fps));
    }
    return fps;
  });
}

struct matrix {
  matrix(size_t count, time value)
      : count_(count), data_(count * count, value) {}

  size_t size() const { return count_; }
  time& operator()(size_t i, size_t j) { return data_[i * count_ + j]; }

  size_t count_;
  std::vector<time> data_;
};

inline void floyd_warshall_serial(matrix& mat) {
  for (auto k = 0UL; k < mat.size(); ++k) {
    for (auto i = 0UL; i < mat.size(); ++i) {
      for (auto j = 0UL; j < mat.size(); ++j) {
        if (mat(i, j) > mat(i, k) + mat(k, j)) {
          mat(i, j) = mat(i, k) + mat(k, j);
        }
      }
    }
  }
}

std::vector<std::pair<uint32_t, uint32_t>> find_components(
    footgraph const& fgraph) {
  std::vector<std::pair<uint32_t, uint32_t>> components(fgraph.size());
  std::generate(begin(components), end(components), [i = 0UL]() mutable {
    return std::pair<uint32_t, uint32_t>{kNoComponent, i++};
  });

  std::stack<uint32_t> stack;  // invariant: stack is empty
  for (auto i = 0UL; i < fgraph.size(); ++i) {
    if (components[i].first != kNoComponent || fgraph[i].empty()) {
      continue;
    }

    stack.emplace(i);
    while (!stack.empty()) {
      auto j = stack.top();
      stack.pop();

      if (components[j].first == i) {
        continue;
      }

      components[j].first = i;
      for (auto const& f : fgraph[j]) {
        if (components[f.to_station_].first != i) {
          stack.push(f.to_station_);
        }
      }
    }
  }

  return components;
}

void process_component(component_it const lb, component_it const ub,
                       footgraph const& fgraph, schedule& sched) {
  if (lb->first == kNoComponent) {
    return;
  }

  auto const size = std::distance(lb, ub);
  if (size == 2) {
    auto idx_a = lb->second;
    auto idx_b = std::next(lb)->second;

    if (!fgraph[idx_a].empty()) {
      utl::verify_silent(
          fgraph[idx_a].size() == 1,
          "invalid size (a): idx_a={}, size={}, data=[{}], idx_b={}, size={}",
          idx_a, fgraph[idx_a].size(), to_str(fgraph[idx_a]), idx_b,
          fgraph[idx_b].size(), to_str(fgraph[idx_b]));
      sched.stations_[idx_a]->outgoing_footpaths_.push_back(
          fgraph[idx_a].front());
      sched.stations_[idx_b]->incoming_footpaths_.push_back(
          fgraph[idx_a].front());
    }
    if (!fgraph[idx_b].empty()) {
      utl::verify_silent(
          fgraph[idx_b].size() == 1,
          "invalid size (a): idx_a={}, size={}, idx_b={}, size={}", idx_a,
          fgraph[idx_a].size(), idx_b, fgraph[idx_b].size());
      sched.stations_[idx_b]->outgoing_footpaths_.push_back(
          fgraph[idx_b].front());
      sched.stations_[idx_a]->incoming_footpaths_.push_back(
          fgraph[idx_b].front());
    }
    return;
  }
  utl::verify(size > 2, "invalid size");

  constexpr auto const kInvalidTime = std::numeric_limits<motis::time>::max();
  matrix mat(size, kInvalidTime);

  for (auto i = 0; i < size; ++i) {
    auto it = lb;
    for (auto const& edge : fgraph[(lb + i)->second]) {  // precond.: sorted!
      while (it != ub && edge.to_station_ != it->second) {
        ++it;
      }
      auto j = std::distance(lb, it);
      mat(i, j) = edge.duration_;
    }
  }

  floyd_warshall_serial(mat);

  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < size; ++j) {
      if (mat(i, j) == kInvalidTime || i == j) {
        continue;
      }

      auto idx_a = std::next(lb, i)->second;
      auto idx_b = std::next(lb, j)->second;

      // each node only in one cluster -> no sync required
      sched.stations_[idx_a]->outgoing_footpaths_.emplace_back(idx_a, idx_b,
                                                               mat(i, j));
      sched.stations_[idx_b]->incoming_footpaths_.emplace_back(idx_a, idx_b,
                                                               mat(i, j));
    }
  }
}

void calc_footpaths(schedule& sched) {
  ml::scoped_timer timer("building transitively closed foot graph");

  auto const fgraph = get_footpath_graph(sched.station_nodes_);

  auto components = find_components(fgraph);
  std::sort(begin(components), end(components));

  std::vector<component_range> ranges;
  utl::equal_ranges_linear(
      components,
      [](auto const& a, auto const& b) { return a.first == b.first; },
      [&](auto lb, auto ub) { ranges.emplace_back(lb, ub); });

  auto const errors = utl::parallel_for(
      ranges,
      [&](auto const& range) {
        process_component(range.first, range.second, fgraph, sched);
      },
      utl::parallel_error_strategy::CONTINUE_EXEC);
  if (!errors.empty()) {
    for (auto const& [idx, ex] : errors) {
      try {
        std::rethrow_exception(ex);
      } catch (std::exception const& e) {
        LOG(logging::error)
            << "foopath error: " << idx << " (" << e.what() << ")";
      }
    }
  }
}

}  // namespace motis::loader
