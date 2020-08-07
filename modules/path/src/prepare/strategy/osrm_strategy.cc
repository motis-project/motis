#include "motis/path/prepare/strategy/osrm_strategy.h"

#include <atomic>

#include "osrm/route_parameters.hpp"

#include "engine/datafacade/internal_datafacade.hpp"  // osrm
#include "engine/plugins/viaroute.hpp"  // osrm
#include "engine/routing_algorithms/multi_target.hpp"  // osrm

#include "util/coordinate.hpp"  // osrm
#include "util/json_container.hpp"  // osrm
#include "util/json_renderer.hpp"  // osrm
#include "util/json_util.hpp"  // osrm

#include "geo/latlng.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/parallel_for.h"
#include "utl/repeat_n.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

using namespace osrm;
using namespace osrm::engine;
using namespace osrm::engine::datafacade;
using namespace osrm::engine::plugins;
using namespace osrm::engine::routing_algorithms;
using namespace osrm::storage;
using namespace osrm::util;
using namespace osrm::util::json;

namespace ml = motis::logging;

namespace motis::path {

constexpr auto kInitialMatchDistance = 10.0;
constexpr auto kFallbackMatchDistance = 100.0;
constexpr auto kMinNodes = 2;
constexpr auto kMaxNodes = 5;

FloatCoordinate make_coord(geo::latlng const& pos) {
  return FloatCoordinate{FloatLongitude{pos.lng_}, FloatLatitude{pos.lat_}};
}

geo::latlng make_latlng(FloatCoordinate const& coord) {
  return {static_cast<double>(coord.lat), static_cast<double>(coord.lon)};
}

struct osrm_strategy::impl {
  impl(strategy_id_t const strategy_id, std::vector<station> const& stations,
       std::string const& path)
      : strategy_id_(strategy_id),
        osrm_data_facade_(
            std::make_unique<InternalDataFacade>(StorageConfig{path})),
        osrm_heaps_(std::make_unique<SearchEngineData>()),
        mt_forward_(osrm_data_facade_.get(), *osrm_heaps_),
        mt_backward_(osrm_data_facade_.get(), *osrm_heaps_),
        via_route_(std::make_unique<ViaRoutePlugin>(*osrm_data_facade_, -1)) {
    ml::scoped_timer t{"load osrm strategy"};

    std::mutex m;
    std::vector<int> node_counts(stations.size());
    utl::parallel_for(
        "process station", stations, 25000, [&, this](auto const& station) {
          std::vector<PhantomNodeWithDistance> nodes_dists;
          auto const find_phantom_nodes = [&](auto const& pos, auto const count,
                                              auto const distance) {
            auto const node_dist =
                osrm_data_facade_->NearestPhantomNodesFromBigComponent(
                    make_coord(pos), count);
            for (auto const& node_with_dist : node_dist) {
              if (node_with_dist.distance <= distance) {
                nodes_dists.push_back(node_with_dist);
              }
            }
          };

          auto const count =
              std::none_of(begin(station.stop_positions_),
                           end(station.stop_positions_),
                           [](auto const& sp) {
                             return sp.categories_.empty() ||
                                    sp.has_category(source_spec::category::BUS);
                           })
                  ? kMaxNodes
                  : kMinNodes;

          for (auto const distance :
               {kInitialMatchDistance, kFallbackMatchDistance}) {
            find_phantom_nodes(station.pos_, count, distance);
            for (auto const& sp : station.stop_positions_) {
              if (!sp.categories_.empty() &&
                  !sp.has_category(source_spec::category::BUS)) {
                continue;
              }
              find_phantom_nodes(sp.pos_, count, distance);
            }

            if (!nodes_dists.empty()) {
              break;
            }
          }

          utl::erase_duplicates(
              nodes_dists,
              [](auto const& lhs, auto const& rhs) {
                return std::tie(lhs.phantom_node.location.lon,
                                lhs.phantom_node.location.lat, lhs.distance) <
                       std::tie(rhs.phantom_node.location.lon,
                                rhs.phantom_node.location.lat, rhs.distance);
              },
              [](auto const& lhs, auto const& rhs) {
                return lhs.phantom_node == rhs.phantom_node;
              });
          auto node_count = nodes_dists.size();

          {
            std::vector<PhantomNodeWithDistance> extra_nodes;
            for (auto& node_with_dist : nodes_dists) {
              auto& pn = node_with_dist.phantom_node;
              if (pn.reverse_segment_id.enabled != 0U &&
                  pn.forward_segment_id.enabled != 0U) {
                pn.forward_segment_id.enabled = 0;
                pn.reverse_segment_id.enabled = 0;

                auto only_forward = node_with_dist;
                only_forward.phantom_node.forward_segment_id.enabled = 1;
                only_forward.phantom_node.reverse_segment_id.enabled = 0;

                pn.forward_segment_id.enabled = 0;
                pn.reverse_segment_id.enabled = 1;

                extra_nodes.push_back(only_forward);
              }
            }
            utl::concat(nodes_dists, extra_nodes);
          }

          auto const lock = std::lock_guard{m};
          node_counts.push_back(node_count);
          stations_to_nodes_[station.id_] =
              utl::to_vec(nodes_dists, [&](auto const& node_with_dist) {
                node_mem_.emplace_back(node_with_dist);
                return node_mem_.size() - 1;
              });
        });

    auto const count = std::accumulate(begin(node_counts), end(node_counts), 0);
    auto const avg = count / stations.size();
    LOG(motis::logging::info) << "osrm node stats";
    LOG(motis::logging::info) << "- stations: " << stations.size();
    LOG(motis::logging::info) << "- nodes: " << count;
    if (!node_counts.empty()) {
      std::sort(begin(node_counts), end(node_counts));
      LOG(motis::logging::info)
          << "- nodes per station: "
          << " avg:" << avg  //
          << " q75:" << node_counts[0.75 * (node_counts.size() - 1)]  //
          << " q90:" << node_counts[0.90 * (node_counts.size() - 1)]  //
          << " q95:" << node_counts[0.95 * (node_counts.size() - 1)];
    }

    for (auto const& [station_id, node_indices] : stations_to_nodes_) {
      stations_to_refs_[station_id] =
          utl::to_vec(node_indices, [&](auto const& idx) -> node_ref {
            return {strategy_id_, idx,
                    make_latlng(node_mem_[idx].phantom_node.location),
                    node_mem_[idx].distance};
          });
    }
  }

  std::vector<node_ref> const& close_nodes(
      std::string const& station_id) const {
    auto const it = stations_to_refs_.find(station_id);
    utl::verify(it != end(stations_to_refs_), "osrm: unknown station!");
    return it->second;
  }

  routing_result_matrix find_routes(std::vector<node_ref> const& from,
                                    std::vector<node_ref> const& to) {
    auto const pair_to_weight = [&](auto const& pair) {
      return pair.first == INVALID_EDGE_WEIGHT
                 ? std::numeric_limits<double>::infinity()
                 : pair.second;
    };

    auto const route = [&](auto const& from_nodes, auto const& to_nodes,
                           bool forward) {
      std::vector<PhantomNode> query_phantoms{PhantomNode{}};
      utl::concat(query_phantoms, to_nodes);

      routing_result_matrix mat{from_nodes.size(), to_nodes.size()};
      for (auto i = 0UL; i < from_nodes.size(); ++i) {
        query_phantoms[0] = from_nodes[i];

        auto const results = forward ? mt_forward_(query_phantoms)
                                     : mt_backward_(query_phantoms);
        utl::verify(static_cast<bool>(results), "osrm routing error!");

        for (auto j = 0UL; j < to_nodes.size(); ++j) {
          auto weight = pair_to_weight(results->at(j));
          if (from_nodes[i] == to_nodes[j]) {
            weight *= 10;
          }
          mat(i, j) = weight;
        }
      }

      if (!forward) {
        mat.transpose();
      }
      return mat;
    };

    auto const from_nodes = utl::to_vec(
        from, [&](auto const& f) { return node_mem_[f.id_].phantom_node; });
    auto const to_nodes = utl::to_vec(
        to, [&](auto const& t) { return node_mem_[t.id_].phantom_node; });
    if (from_nodes.size() <= to_nodes.size()) {
      return route(from_nodes, to_nodes, true);
    } else {
      return route(to_nodes, from_nodes, false);
    }
  }

  osm_path get_path(node_ref const& from, node_ref const& to) const {
    utl::verify(from.strategy_id_ == strategy_id_,
                "osrm bad 'from' strategy_id");
    utl::verify(to.strategy_id_ == strategy_id_, "osrm bad 'to' strategy_id");

    RouteParameters params;
    params.geometries = RouteParameters::GeometriesType::CoordVec1D;
    params.overview = RouteParameters::OverviewType::Full;
    params.osm_node_ids = true;

    params.coordinates.push_back(node_mem_[from.id_].phantom_node.location);
    params.coordinates.push_back(node_mem_[to.id_].phantom_node.location);

    Object result;
    auto const status = via_route_->HandleRequest(params, result);
    auto& all_routes = result.values["routes"];

    utl::verify(status == Status::Ok && !all_routes.get<Array>().values.empty(),
                "no path found in osrm_strategy");

    auto& route = get(all_routes, 0U);

    auto const& osrm_polyline = get(route, "geometry").get<Array>().values;
    mcd::vector<geo::latlng> polyline;
    polyline.reserve(osrm_polyline.size() / 2);
    for (auto i = 0U; i < osrm_polyline.size(); i += 2) {
      polyline.emplace_back(osrm_polyline[i].template get<Number>().value,
                            osrm_polyline[i + 1].template get<Number>().value);
    }

    static_assert(sizeof(long long) == 8);  // NOLINT
    auto osm_node_ids =
        mcd::to_vec(get(route, "osm_node_ids").get<Array>().values,
                    [](auto&& e) -> int64_t {
                      return std::stoll(e.template get<String>().value);
                    });

    utl::verify(!osm_node_ids.empty(), "osrm_strategy: empty osm_node_ids");

    // XXX remove this after OSRM-backend update
    osm_node_ids.front() = kPathUnknownNodeId;
    osm_node_ids.back() = kPathUnknownNodeId;

    return osm_path{std::move(polyline), std::move(osm_node_ids)};
  }

  strategy_id_t strategy_id_;

  std::unique_ptr<InternalDataFacade> osrm_data_facade_;

  std::unique_ptr<SearchEngineData> osrm_heaps_;
  MultiTargetRouting<BaseDataFacade, true> mt_forward_;
  MultiTargetRouting<BaseDataFacade, false> mt_backward_;

  std::unique_ptr<ViaRoutePlugin> via_route_;

  std::vector<PhantomNodeWithDistance> node_mem_;
  mcd::hash_map<std::string, std::vector<size_t>> stations_to_nodes_;
  mcd::hash_map<std::string, std::vector<node_ref>> stations_to_refs_;
};

osrm_strategy::osrm_strategy(strategy_id_t strategy_id, source_spec spec,
                             std::vector<station> const& stations,
                             std::string const& osrm_path)
    : routing_strategy(strategy_id, spec),
      impl_(std::make_unique<osrm_strategy::impl>(strategy_id, stations,
                                                  osrm_path)) {}
osrm_strategy::~osrm_strategy() = default;

std::vector<node_ref> const& osrm_strategy::close_nodes(
    std::string const& station_id) const {
  return impl_->close_nodes(station_id);
}

routing_result_matrix osrm_strategy::find_routes(
    std::vector<node_ref> const& from, std::vector<node_ref> const& to) const {
  return impl_->find_routes(from, to);
}

osm_path osrm_strategy::get_path(node_ref const& from,
                                 node_ref const& to) const {
  return impl_->get_path(from, to);
}

}  // namespace motis::path
