#include "motis/loader/build_footpaths.h"

#include <optional>
#include <stack>

#include "geo/latlng.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/parallel_for.h"
#include "utl/verify.h"

#include "motis/core/common/floyd_warshall.h"
#include "motis/core/common/logging.h"
#include "motis/core/schedule/price.h"
#include "motis/loader/filter/local_stations.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace f = flatbuffers64;
namespace ml = motis::logging;

namespace motis::loader {

constexpr auto const kAdjustedMaxDuration = 15;  // [minutes]

constexpr auto kNoComponent = std::numeric_limits<uint32_t>::max();

// station_idx -> [footpath, ...]
using footgraph = std::vector<std::vector<footpath>>;

// (component, original station_idx)
using component_vec = std::vector<std::pair<uint32_t, uint32_t>>;
using component_it = component_vec::iterator;
using component_range = std::pair<component_it, component_it>;

struct footpath_builder {

  footpath_builder(
      schedule& sched, loader_options const& opt,
      mcd::hash_map<Station const*, station_node*> const& station_nodes)
      : sched_{sched}, opt_{opt}, station_nodes_{station_nodes} {}

  void add_footpaths(f::Vector<f::Offset<Footpath>> const* footpaths) {
    auto skipped = 0U;
    for (auto const& footpath : *footpaths) {
      auto const get_station = [&](char const* tag, Station const* s) {
        auto const it = station_nodes_.find(s);
        utl::verify(it != end(station_nodes_),
                    "footpath {} node not found {} [{}] ", tag,
                    footpath->from()->name()->c_str(),
                    footpath->from()->id()->c_str());
        return it->second;
      };

      if (skip_station(footpath->from()) || skip_station(footpath->to())) {
        continue;
      }

      auto duration = static_cast<int32_t>(footpath->duration());
      auto const from_node = get_station("from", footpath->from());
      auto const to_node = get_station("to", footpath->to());
      auto& from_station = sched_.stations_.at(from_node->id_);
      auto& to_station = sched_.stations_.at(to_node->id_);

      if (from_node == to_node) {
        LOG(ml::warn) << "Footpath loop at station " << from_station->eva_nr_
                      << " ignored";
        continue;
      }

      if (opt_.adjust_footpaths_) {
        duration = std::max({from_station->transfer_time_,
                             to_station->transfer_time_, duration});
        auto const distance = get_distance(*from_station, *to_station) * 1000;

        auto adjusted_duration = adjust_footpath_duration(duration, distance);
        if (!adjusted_duration.has_value()) {
          continue;
        }
        duration = *adjusted_duration;
      }

      from_station->equivalent_.emplace_back(to_station.get());
      add_foot_edge_pair(from_node, to_node, duration);
    }
    LOG(ml::info) << "Skipped " << skipped
                  << " footpaths connecting stations with no events";
  }

  static std::optional<int> adjust_footpath_duration(int duration,
                                                     int const distance) {
    auto const max_distance_adjust = duration * 60 * WALK_SPEED;
    auto const max_distance = 2 * duration * 60 * WALK_SPEED;

    if (distance > max_distance) {
      return {};
    } else if (distance > max_distance_adjust) {
      duration = std::round(distance / (60 * WALK_SPEED));
    }

    if (duration > kAdjustedMaxDuration) {
      return {};
    }

    return {duration};
  }

  void add_foot_edge_pair(station_node* from_sn, station_node* to_sn,
                          uint16_t const duration) {
    auto* from_fn = get_or_create_foot_node(from_sn);
    auto* to_fn = get_or_create_foot_node(to_sn);

    // FROM_FOOT -(FWD)-> TO_STATION
    from_fn->edges_.emplace_back(make_fwd_edge(from_fn, to_sn, duration));

    // FROM_STATION -(BWD)-> TO_FOOT
    from_sn->edges_.emplace_back(make_bwd_edge(from_sn, to_fn, duration));
  }

  node* get_or_create_foot_node(node* n) {
    station_node* sn = n->get_station();
    if (sn->foot_node_ != nullptr) {
      return sn->foot_node_.get();
    }

    // Create the foot node.
    auto foot_node = mcd::make_unique<node>();
    foot_node->station_node_ = sn;
    foot_node->id_ = sched_.node_count_++;

    // STATION_NODE -(FWD_EDGE)-> FOOT_NODE
    sn->edges_.emplace_back(make_fwd_edge(sn, foot_node.get()));

    // FOOT_NODE -(BWD_EDGE)-> STATION_NODE
    foot_node->edges_.emplace_back(make_bwd_edge(foot_node.get(), sn));

    // ROUTE_NODE -(AFTER_TRAIN_FWD)-> STATION_NODE
    sn->for_each_route_node([&](auto&& route_node) {
      // check whether it is allowed to transfer at the route-node
      // we do this by checking, whether it has an edge to the station
      for (auto const& e : route_node->edges_) {
        if (e.to_ == sn && e.type() != edge::INVALID_EDGE) {
          // the foot-edge may only be used
          // if a train was used beforewards when
          // trying to use it from a route node
          route_node->edges_.push_back(
              make_after_train_fwd_edge(route_node, foot_node.get(), 0, true));
          break;
        }
      }
    });

    // STATION_NODE -(AFTER_TRAIN_BWD)-> ROUTE_NODE
    for (auto const& e : sn->edges_) {
      if (e.to_->is_route_node() && e.type() != edge::INVALID_EDGE) {
        foot_node->edges_.emplace_back(
            make_after_train_bwd_edge(foot_node.get(), e.to_, 0, true));
      }
    }

    sn->foot_node_ = std::move(foot_node);
    return sn->foot_node_.get();
  }

  void equivalences_to_footpaths() {
    for (auto const& from_s : sched_.stations_) {
      auto* from_sn = sched_.station_nodes_.at(from_s->index_).get();

      for (auto const& to_s : from_s->equivalent_) {
        if (from_s->source_schedule_ == to_s->source_schedule_) {
          continue;  // no footpaths for schedule-defined meta stations
        }

        auto* to_sn = sched_.station_nodes_.at(to_s->index_).get();

        auto const distance =
            geo::distance(geo::latlng{from_s->lat(), from_s->lng()},
                          geo::latlng{to_s->lat(), to_s->lng()});
        auto const duration =
            std::max({from_s->transfer_time_, to_s->transfer_time_,
                      static_cast<int32_t>(std::round(
                          static_cast<double>(distance) / (60 * WALK_SPEED)))});
        add_foot_edge_pair(from_sn, to_sn, duration);
      }
    }
  }

  void transitivize_footpaths() {
    ml::scoped_timer timer("building transitively closed foot graph");

    auto const fgraph = get_footpath_graph();

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
          process_component(range.first, range.second, fgraph);
        },
        utl::parallel_error_strategy::CONTINUE_EXEC);
    if (!errors.empty()) {
      for (auto const& [idx, ex] : errors) {
        try {
          std::rethrow_exception(ex);
        } catch (std::exception const& e) {
          LOG(ml::error) << "footpath error: " << idx << " (" << e.what()
                         << ")";
        }
      }
    }
  }

  void make_station_equivalents_unique() {
    for (auto& s : sched_.stations_) {
      if (s->equivalent_.size() <= 1) {
        continue;
      }

      utl::erase_duplicates(
          s->equivalent_, begin(s->equivalent_) + 1, end(s->equivalent_),
          [](auto const& a, auto const& b) { return a->index_ < b->index_; },
          [](auto const& a, auto const& b) { return a->index_ == b->index_; });

      s->equivalent_.erase(
          std::remove_if(begin(s->equivalent_) + 1, end(s->equivalent_),
                         [&s](auto const& equivalent) {
                           return equivalent->index_ == s->index_;
                         }),
          end(s->equivalent_));
    }
  }

  footgraph get_footpath_graph() const {
    return utl::to_vec(sched_.station_nodes_, [](auto const& station_node) {
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

  static std::vector<std::pair<uint32_t, uint32_t>> find_components(
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
                         footgraph const& fgraph) {
    if (lb->first == kNoComponent) {
      return;
    }

    auto const size = std::distance(lb, ub);
    if (size == 2) {
      auto idx_a = lb->second;
      auto idx_b = std::next(lb)->second;

      if (!fgraph[idx_a].empty()) {
        utl::verify_silent(fgraph[idx_a].size() == 1,
                           "invalid size (a): idx_a={}, size={}, data=[{}], "
                           "idx_b={}, size = {} ",
                           idx_a, fgraph[idx_a].size(), to_str(fgraph[idx_a]),
                           idx_b, fgraph[idx_b].size(), to_str(fgraph[idx_b]));
        sched_.stations_[idx_a]->outgoing_footpaths_.push_back(
            fgraph[idx_a].front());
        sched_.stations_[idx_b]->incoming_footpaths_.push_back(
            fgraph[idx_a].front());
      }
      if (!fgraph[idx_b].empty()) {
        utl::verify_silent(
            fgraph[idx_b].size() == 1,
            "invalid size (a): idx_a={}, size={}, idx_b={}, size={}", idx_a,
            fgraph[idx_a].size(), idx_b, fgraph[idx_b].size());
        sched_.stations_[idx_b]->outgoing_footpaths_.push_back(
            fgraph[idx_b].front());
        sched_.stations_[idx_a]->incoming_footpaths_.push_back(
            fgraph[idx_b].front());
      }
      return;
    }
    utl::verify(size > 2, "invalid size");

    constexpr auto const kInvalidTime = std::numeric_limits<motis::time>::max();
    auto mat = make_flat_matrix<motis::time>(size, kInvalidTime);

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

    floyd_warshall(mat);

    for (auto i = 0; i < size; ++i) {
      for (auto j = 0; j < size; ++j) {
        if (mat(i, j) == kInvalidTime || i == j) {
          continue;
        }

        auto idx_a = std::next(lb, i)->second;
        auto idx_b = std::next(lb, j)->second;

        // each node only in one cluster -> no sync required
        sched_.stations_[idx_a]->outgoing_footpaths_.emplace_back(idx_a, idx_b,
                                                                  mat(i, j));
        sched_.stations_[idx_b]->incoming_footpaths_.emplace_back(idx_a, idx_b,
                                                                  mat(i, j));
      }
    }
  }

  static std::string to_str(std::vector<footpath> const& footpaths) {
    std::stringstream s;
    for (auto const& [i, fp] : utl::enumerate(footpaths)) {
      s << (i == 0 ? "" : ", ") << fp.from_station_ << "-" << fp.to_station_
        << "-" << fp.duration_ << "min";
    }
    return s.str();
  }

  inline bool skip_station(Station const* station) {
    return opt_.no_local_transport_ && is_local_station(station);
  }

  schedule& sched_;
  loader_options const& opt_;
  mcd::hash_map<Station const*, station_node*> const& station_nodes_;
};

void build_footpaths(
    schedule& sched, loader_options const& opt,
    mcd::hash_map<Station const*, station_node*> const& station_nodes,
    std::vector<Schedule const*> const& fbs_schedules) {
  footpath_builder b{sched, opt, station_nodes};

  for (auto const* fbs_schedule : fbs_schedules) {
    b.add_footpaths(fbs_schedule->footpaths());
  }

  b.equivalences_to_footpaths();
  b.make_station_equivalents_unique();

  if (opt.expand_footpaths_) {
    // progress_tracker.status("Expand Footpaths").out_bounds(95, 96);
    b.transitivize_footpaths();
  }
}

}  // namespace motis::loader
