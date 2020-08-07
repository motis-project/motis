#pragma once

#include "geo/point_rtree.h"

#include "utl/concat.h"
#include "utl/equal_ranges_linear.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/path/prepare/tuning_parameters.h"

#include "motis/path/prepare/osm/osm_graph.h"
#include "motis/path/prepare/osm/osm_graph_builder.h"
#include "motis/path/prepare/osm/osm_graph_contractor.h"
#include "motis/path/prepare/osm/osm_router.h"

#include "motis/path/prepare/strategy/routing_strategy.h"

namespace ml = motis::logging;

namespace motis::path {

// TODO(sebastian) penalize sharp "Z"-Style turns in rail networks
struct osm_strategy : public routing_strategy {
  osm_strategy(strategy_id_t strategy_id, source_spec spec,
               station_index const& station_idx,
               mcd::vector<mcd::vector<osm_way>> const& components)
      : routing_strategy(strategy_id, spec) {
    osm_graph_builder{graph_, spec, station_idx}.build_graph(components);
    print_osm_graph_stats(spec, graph_);

    contracted_ = contract_graph(graph_);
    print_osm_graph_stats(spec, contracted_);

    utl::equal_ranges_linear(
        graph_.node_station_links_,
        [](auto const& lhs, auto const& rhs) {
          return lhs.station_id_ == rhs.station_id_;
        },
        [&](auto lb, auto ub) {
          auto refs = utl::to_vec(lb, ub, [&](auto const& link) {
            return node_ref{this->strategy_id_, link.node_idx_,
                            graph_.nodes_[link.node_idx_]->pos_,
                            link.distance_};
          });

          for (auto i = 0UL; i < refs.size(); ++i) {
            auto& ni = *graph_.nodes_[refs[i].id_];

            if (refs[i].node_equiv_ != kInvalidNodeEquiv) {
              continue;
            }

            refs[i].node_equiv_ = i;
            for (auto j = i + 1UL; j < refs.size(); ++j) {
              auto& nj = *graph_.nodes_[refs[j].id_];

              bool eq = false;
              if (ni.osm_id_ == -1 && nj.osm_id_ == -1) {
                eq = ni.pos_ == nj.pos_;
              } else {
                eq = ni.osm_id_ == nj.osm_id_;
              }

              if (eq) {
                refs[j].node_equiv_ = i;
              }
            }
          }

          stations_to_refs_[lb->station_id_] = std::move(refs);
        });
    LOG(ml::info) << "- mapped stations: " << stations_to_refs_.size();
  }

  ~osm_strategy() override = default;
  osm_strategy(osm_strategy const&) noexcept = delete;
  osm_strategy& operator=(osm_strategy const&) noexcept = delete;
  osm_strategy(osm_strategy&&) noexcept = delete;
  osm_strategy& operator=(osm_strategy&&) noexcept = delete;

  std::vector<node_ref> const& close_nodes(
      std::string const& station_id) const override {
    auto const it = stations_to_refs_.find(station_id);
    if (it == end(stations_to_refs_)) {
      return unknown_station_refs_;
    }
    return it->second;
  }

  routing_result_matrix find_routes(
      std::vector<node_ref> const& from,
      std::vector<node_ref> const& to) const override {
    routing_result_matrix mat{from.size(), to.size()};

    auto const to_ids = utl::to_vec(to, [](auto&& t) { return t.id_; });
    for (auto i = 0UL; i < from.size(); ++i) {
      auto dists = shortest_path_distances(contracted_, from[i].id_, to_ids);

      auto const factor = source_spec_.router_ == source_spec::router::OSM_REL
                              ? kOsmRelationBonusFactor
                              : 1;
      for (auto j = 0UL; j < to.size(); ++j) {
        mat(i, j) = dists.at(j) * factor;
      }
    }

    {  // only keep best of equivalent node pairs
      struct eq_best {
        size_t eq_from_, eq_to_;
        size_t best_from_, best_to_;
      };
      std::vector<eq_best> eq_bests;
      for (auto i = 0UL; i < from.size(); ++i) {
        for (auto j = 0UL; j < to.size(); ++j) {
          if (i == j) {
            continue;
          }

          auto const& curr_eq_from = from[i].node_equiv_;
          auto const& curr_eq_to = to[j].node_equiv_;

          auto it =
              std::find_if(begin(eq_bests), end(eq_bests), [&](auto const& eb) {
                return eb.eq_from_ == curr_eq_from && eb.eq_to_ == curr_eq_to;
              });
          if (it == end(eq_bests)) {
            eq_bests.push_back({curr_eq_from, curr_eq_to, i, j});
            continue;
          }

          auto& curr = mat(i, j);
          auto& best = mat(it->best_from_, it->best_to_);

          if (curr < best) {
            it->best_from_ = i;
            it->best_to_ = j;
            best = std::numeric_limits<double>::infinity();
          } else {
            curr = std::numeric_limits<double>::infinity();
          }
        }
      }
    }

    return mat;
  }

  osm_path get_path(node_ref const& from, node_ref const& to) const override {
    utl::verify(from.strategy_id_ == strategy_id_,
                "osm bad 'from' strategy_id");
    utl::verify(to.strategy_id_ == strategy_id_, "osm bad 'to' strategy_id");

    if (from.id_ == to.id_) {
      auto const& node = graph_.nodes_.at(from.id_);
      return osm_path{{node->pos_, node->pos_}, {node->osm_id_, node->osm_id_}};
    }

    auto const path = shortest_paths(graph_, from.id_, {to.id_}, true).at(0);
    utl::verify(!path.empty(), "empty shortest path in osm_strategy");

    osm_path result;
    for (auto const& edge : path) {
      utl::verify(edge != nullptr, "osm (get_path) found invalid edge");
      auto const& path = graph_.paths_[edge->polyline_idx_];
      if (edge->is_forward()) {
        result.append(path);
      } else {
        osm_path mutable_path{path};  // XXX make this faster
        mutable_path.reverse();
        result.append(mutable_path);
      }
    }
    return result;
  }

  osm_graph graph_, contracted_;

  std::vector<node_ref> unknown_station_refs_;
  mcd::hash_map<std::string, std::vector<node_ref>> stations_to_refs_;
};

}  // namespace motis::path
