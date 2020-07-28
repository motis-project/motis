#include "motis/path/prepare/osm/osm_graph_builder.h"

#include <algorithm>

#include "boost/range/irange.hpp"

#include "geo/xyz.h"

#include "utl/concat.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
#include "utl/repeat_n.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/tuning_parameters.h"

using namespace motis::logging;
using namespace geo;

namespace motis::path {

using osm_idx_t = int64_t;
using node_idx_t = size_t;
using station_idx_t = size_t;
using offset_t = size_t;

void osm_graph_builder::build_graph(
    mcd::vector<mcd::vector<osm_way>> const& components) {
  constexpr auto const kSmallComponentLimit = 10000;
  utl::parallel_for("add small component", components, 1000,
                    [&](auto const& c) {
                      if (c.size() < kSmallComponentLimit) {
                        add_component(c);
                      }
                    });
  LOG(info) << "add large components";
  for (auto const& c : components) {
    if (c.size() < kSmallComponentLimit) {
      continue;
    }
    if (osm_phantom_builder pb{station_idx_, c};
        !pb.matched_stations_.empty()) {
      auto const& matched_stations = pb.matched_stations_;

      auto const task_count =
          std::min<size_t>(std::max<size_t>(matched_stations.size() / 1000, 1),
                           2 * std::thread::hardware_concurrency());
      auto const batch_size = matched_stations.size() / task_count;
      utl::verify(batch_size != 0, "osm_graph_builder: empty batch size");

      utl::parallel_for(boost::irange(task_count), [&](auto const batch_idx) {
        auto const lb = batch_idx * batch_size;
        auto const ub = (batch_idx + 1 != task_count)
                            ? (batch_idx + 1) * batch_size
                            : matched_stations.size();
        for (auto i = lb; i < ub; ++i) {
          pb.build_osm_phantoms(matched_stations[i]);
        }
      });
      pb.finalize();

      if (pb.n_phantoms_.size() + pb.e_phantoms_.size() > 0) {
        add_component(c, pb.n_phantoms_, pb.e_phantoms_);
      }
    }
  }

  std::sort(begin(graph_.node_station_links_), end(graph_.node_station_links_),
            [](auto const& lhs, auto const& rhs) {
              return std::tie(lhs.station_id_, lhs.node_idx_, lhs.distance_) <
                     std::tie(rhs.station_id_, rhs.node_idx_, rhs.distance_);
            });

  utl::equal_ranges_linear(
      graph_.nodes_,
      [](auto const& lhs, auto const& rhs) {
        return lhs->component_id_ == rhs->component_id_;
      },
      [&](auto const lb, auto const ub) {
        utl::verify((*lb)->component_id_ == graph_.component_offsets_.size(),
                    "unexpected component");
        graph_.component_offsets_.emplace_back(
            std::distance(begin(graph_.nodes_), lb), std::distance(lb, ub));
      });
}

void osm_graph_builder::add_component(mcd::vector<osm_way> const& osm_ways) {
  if (osm_phantom_builder pb{station_idx_, osm_ways};
      !pb.matched_stations_.empty()) {
    for (auto const* station : pb.matched_stations_) {
      pb.build_osm_phantoms(station);
    }
    pb.finalize();

    if (pb.n_phantoms_.size() + pb.e_phantoms_.size() > 0) {
      add_component(osm_ways, pb.n_phantoms_, pb.e_phantoms_);
    }
  }
}

void osm_graph_builder::add_component(
    mcd::vector<osm_way> const& osm_ways,
    std::vector<osm_node_phantom_match> const& n_phantoms,
    std::vector<osm_edge_phantom_match> const& e_phantoms) {
  auto const lock = std::lock_guard{mutex_};
  auto component = graph_.components_++;

  mcd::hash_map<osm_idx_t, osm_node*> node_map;
  auto const make_station_links = [&](auto node_idx, auto station_dists) {
    utl::erase_duplicates(
        station_dists, [](auto const& a, auto const& b) { return a < b; },
        [](auto const& a, auto const& b) { return a.first == b.first; });
    for (auto const& [station_id, distance] : station_dists) {
      graph_.node_station_links_.emplace_back(station_id, node_idx, distance);
    }
  };

  auto const make_osm_node = [&](osm_idx_t const id, geo::latlng const& pos) {
    return utl::get_or_create(node_map, id, [&] {
      auto const node_idx = graph_.nodes_.size();
      auto& node = graph_.nodes_.emplace_back(
          std::make_unique<osm_node>(node_idx, component, id, pos));

      std::vector<std::pair<std::string, double>> links;
      for (auto it = std::lower_bound(begin(n_phantoms), end(n_phantoms), id,
                                      [](auto const&lhs, auto const&rhs) {
                                        return lhs.phantom_.id_ < rhs;
                                      });
           it != end(n_phantoms) && it->phantom_.id_ == id; ++it) {
        links.emplace_back(it->station_->id_, it->distance_);
      }
      make_station_links(node_idx, links);

      return node.get();
    });
  };

  auto const make_way_node = [&](auto lb, auto ub) {
    auto const node_idx = graph_.nodes_.size();
    auto& node = graph_.nodes_.emplace_back(
        std::make_unique<osm_node>(node_idx, component, -1, lb->pos_));

    make_station_links(
        node_idx, utl::to_vec(lb, ub, [](auto const& p) {
          return std::pair<std::string, double>{p.station_->id_, p.distance_};
        }));

    return node.get();
  };

  auto const make_path = [this](auto const& way, size_t const from,
                                size_t const to, auto const& left_opt,
                                auto const& right_opt) {
    if (from == to) {
      osm_path path{2};
      if (left_opt) {
        path.polyline_.push_back(*left_opt);
        path.osm_node_ids_.push_back(kPathUnknownNodeId);
      } else {
        path.polyline_.push_back(way.path_.polyline_.at(from));
        path.osm_node_ids_.push_back(way.path_.osm_node_ids_.at(from));
      }
      if (right_opt) {
        path.polyline_.push_back(*right_opt);
        path.osm_node_ids_.push_back(kPathUnknownNodeId);
      } else {
        path.polyline_.push_back(way.path_.polyline_.at(from + 1));
        path.osm_node_ids_.push_back(way.path_.osm_node_ids_.at(from + 1));
      }
      graph_.paths_.emplace_back(std::move(path));
    } else {
      graph_.paths_.emplace_back(
          way.path_.partial_replaced_padded(from, to + 1, left_opt, right_opt));
    }

    return graph_.paths_.size() - 1;
  };

  auto const make_edges = [this](auto from, auto to, auto const path_idx,
                                 source_bits const sb) {
    auto dist = 0.;
    for (auto const& [a, b] :
         utl::pairwise(graph_.paths_[path_idx].polyline_)) {
      // dist via xyz coordinates for consistency with a-star heuristic
      dist += geo::haversine_distance(geo::xyz{a}, geo::xyz{b});
    }

    auto const penalty_factor = get_penalty_factor(sb);
    if (penalty_factor == std::numeric_limits<double>::infinity()) {
      return;
    }

    dist *= penalty_factor;

    from->edges_.emplace_back(path_idx, true, dist, from, to);
    if ((sb & source_bits::ONEWAY) != source_bits::ONEWAY) {
      to->edges_.emplace_back(path_idx, false, dist, to, from);
    }
  };

  auto const foreach_new_way_node = [&](auto const way_idx, auto&& fn) {
    auto const dist_less_than = [&](auto from, auto to, auto max_distance) {
      double dist = to->along_track_dist_ - from->along_track_dist_;
      for (auto offset = from->phantom_.offset_; offset < to->phantom_.offset_;
           ++offset) {
        if (dist >= max_distance) {
          return false;
        }

        dist += geo::distance(osm_ways[way_idx].path_.polyline_[offset],
                              osm_ways[way_idx].path_.polyline_[offset + 1]);
      }

      return dist < max_distance;
    };

    for (auto it = std::lower_bound(begin(e_phantoms), end(e_phantoms), way_idx,
                                    [](auto const&a, auto const&b) {
                                      return a.phantom_.way_idx_ < b;
                                    });
         it != end(e_phantoms) && it->phantom_.way_idx_ == way_idx;) {
      auto lb = it;
      while (it != end(e_phantoms) && it->phantom_.way_idx_ == way_idx &&
             dist_less_than(lb, it, 5)) {
        ++it;
      }
      fn(lb, it);
    }
  };

  for (auto i = 0UL; i < osm_ways.size(); ++i) {
    auto const& way = osm_ways[i];

    auto prev_node = make_osm_node(way.from(), way.path_.polyline_.front());
    auto prev_offset = 0;
    std::optional<geo::latlng> prev_coord;  // coord was "invented" by phantom

    foreach_new_way_node(i, [&](auto lb, auto ub) {
      auto* curr_node = make_way_node(lb, ub);
      auto curr_offset = lb->phantom_.offset_;

      size_t path_idx = 0;
      if (lb->eq_from_) {
        utl::verify(curr_offset > 0, "have edge_phantom for start");
        path_idx = make_path(way, prev_offset, curr_offset, prev_coord,
                             std::optional<geo::latlng>{});
        prev_coord = {};

      } else if (lb->eq_to_) {
        utl::verify(curr_offset < way.path_.size() - 2,
                    "have edge_phantom for end");
        path_idx = make_path(way, prev_offset, ++curr_offset, prev_coord,
                             std::optional<geo::latlng>{});
        prev_coord = {};

      } else {
        path_idx = make_path(way, prev_offset, curr_offset, prev_coord,
                             std::optional<geo::latlng>{lb->pos_});
        prev_coord = {lb->pos_};
      }

      make_edges(prev_node, curr_node, path_idx, way.source_bits_);

      prev_node = curr_node;
      prev_offset = curr_offset;
    });

    utl::verify(prev_offset < way.path_.size() - 1, "prev_offset reached end!");
    auto const path_idx = make_path(way, prev_offset, way.path_.size() - 1,
                                    prev_coord, std::optional<geo::latlng>{});
    auto curr_node = make_osm_node(way.to(), way.path_.polyline_.back());

    make_edges(prev_node, curr_node, path_idx, way.source_bits_);
  }
}

double osm_graph_builder::get_penalty_factor(source_bits const sb) const {
  return 1.;

  if (source_spec_.router_ == source_spec::router::OSM_REL) {
    return 1.;
  }

  if (source_spec_.category_ == source_spec::category::SHIP &&
      (sb & source_bits::SHIP) != source_bits::SHIP) {
    return std::numeric_limits<double>::infinity();
  }

  auto const unlikely_penalty =
      (sb & source_bits::UNLIKELY) == source_bits::UNLIKELY
          ? kUnlikelyPenaltyFactor
          : 1.;
  auto const very_unlikely_penalty =
      (sb & source_bits::VERY_UNLIKELY) == source_bits::VERY_UNLIKELY
          ? kVeryUnlikelyPenaltyFactor
          : 1.;
  auto const extra_penalty = std::max(unlikely_penalty, very_unlikely_penalty);

  auto const match = [&](auto const expected_cat, auto const expected_sb) {
    return source_spec_.category_ == expected_cat &&
           (sb & expected_sb) == expected_sb;
  };

  if (source_spec_.category_ == source_spec::category::UNKNOWN ||
      source_spec_.category_ == source_spec::category::MULTI ||
      source_spec_.category_ == source_spec::category::SHIP ||
      match(source_spec::category::BUS, source_bits::BUS) ||
      match(source_spec::category::TRAM, source_bits::TRAM) ||
      match(source_spec::category::RAIL, source_bits::RAIL) ||
      match(source_spec::category::SUBWAY, source_bits::SUBWAY)) {
    return extra_penalty;
  } else if (match(source_spec::category::BUS, source_bits::TRAM) ||
             match(source_spec::category::TRAM, source_bits::BUS) ||
             match(source_spec::category::TRAM, source_bits::LIGHT_RAIL) ||
             match(source_spec::category::RAIL, source_bits::LIGHT_RAIL) ||
             match(source_spec::category::SUBWAY, source_bits::LIGHT_RAIL)) {
    return kMinorMismatchPenaltyFactor * extra_penalty;
  } else {
    return kMajorMismatchPenaltyFactor * extra_penalty;
  }
}

}  // namespace motis::path
