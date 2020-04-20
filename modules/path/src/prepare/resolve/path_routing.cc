#include "motis/path/prepare/resolve/path_routing.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm/osm_cache.h"
#include "motis/path/prepare/osm/parse_network.h"
#include "motis/path/prepare/osm/parse_relations.h"

#include "motis/path/prepare/strategy/osm_strategy.h"
#include "motis/path/prepare/strategy/osrm_strategy.h"
#include "motis/path/prepare/strategy/stub_strategy.h"

namespace fs = boost::filesystem;
namespace ml = motis::logging;

namespace motis::path {

struct path_routing::strategies {
  std::unique_ptr<osrm_strategy> osrm_;

  std::unique_ptr<osm_strategy> net_rail_;
  std::unique_ptr<osm_strategy> net_sub_;
  std::unique_ptr<osm_strategy> net_tram_;

  std::unique_ptr<osm_strategy> rel_rail_;
  std::unique_ptr<osm_strategy> rel_bus_;
  std::unique_ptr<osm_strategy> rel_sub_;
  std::unique_ptr<osm_strategy> rel_tram_;

  std::unique_ptr<stub_strategy> stub_;

  std::vector<routing_strategy*> rail_strategies_;
  std::vector<routing_strategy*> bus_strategies_;
  std::vector<routing_strategy*> subway_strategies_;
  std::vector<routing_strategy*> tram_strategies_;
};

path_routing::path_routing()
    : strategies_(std::make_unique<path_routing::strategies>()) {}
path_routing::~path_routing() = default;

std::vector<routing_strategy*> const& path_routing::strategies_for(
    source_spec::category const cat) const {
  switch (cat) {
    case source_spec::category::BUS: return strategies_->bus_strategies_;
    case source_spec::category::SUBWAY: return strategies_->subway_strategies_;
    case source_spec::category::TRAM: return strategies_->tram_strategies_;
    case source_spec::category::RAIL: return strategies_->rail_strategies_;
    default: throw utl::fail("no strategies for this category");
  }
}

routing_strategy* path_routing::get_stub_strategy() const {
  return strategies_->stub_.get();
}

std::string relation_cache_name(source_spec::category const& category) {
  switch (category) {
    case source_spec::category::BUS: return "pathcache.bus.relation.raw";
    case source_spec::category::SUBWAY: return "pathcache.subway.relation.raw";
    case source_spec::category::TRAM: return "pathcache.tram.relation.raw";
    case source_spec::category::RAIL: return "pathcache.rail.relation.raw";
    default: throw utl::fail("relation_cache: unknown category!");
  }
}

std::string network_cache_name(source_spec::category const& category) {
  switch (category) {
    case source_spec::category::SUBWAY: return "pathcache.subway.network.raw";
    case source_spec::category::TRAM: return "pathcache.tram.network.raw";
    case source_spec::category::RAIL: return "pathcache.rail.network.raw";
    default: throw utl::fail("graph_cache: unknown category!");
  }
}

template <typename Func>
std::vector<std::vector<osm_way>> load_or_parse(std::string const& fname,
                                                Func&& func) {
  std::vector<std::vector<osm_way>> ways;
  if (fs::is_regular_file(fname)) {
    ml::scoped_timer t{std::string{"load osm from cache "}.append(fname)};
    ways = load_osm_ways(fname);
  } else {
    ml::scoped_timer t{std::string{"parse osm and make cache "}.append(fname)};
    ways = func();
    store_osm_ways(fname, ways);
  }
  return ways;
}

path_routing make_path_routing(station_index const& station_idx,
                               std::string const& osm_path,
                               std::string const& osrm_path) {
  strategy_id_t id = 0;

  auto const load_osm_rel = [&](source_spec::category const cat) {
    auto const fname = relation_cache_name(cat);
    auto const components =
        load_or_parse(fname, [&] { return parse_relations(osm_path, cat); });

    ml::scoped_timer t{std::string{"make osm_strategy REL "}.append(fname)};
    return std::make_unique<osm_strategy>(
        id++, source_spec{cat, source_spec::router::OSM_REL}, station_idx,
        components);
  };
  auto const load_osm_net = [&](source_spec::category const cat) {
    auto const fname = network_cache_name(cat);
    auto const components =
        load_or_parse(fname, [&] { return parse_network(osm_path, cat); });

    ml::scoped_timer t{std::string{"make osm_strategy NET "}.append(fname)};
    return std::make_unique<osm_strategy>(
        id++, source_spec{cat, source_spec::router::OSM_NET}, station_idx,
        components);
  };

  path_routing r;

  r.strategies_->stub_ = std::make_unique<stub_strategy>(
      id++,
      source_spec{source_spec::category::MULTI, source_spec::router::STUB},
      station_idx.stations_);

  r.strategies_->osrm_ = std::make_unique<osrm_strategy>(
      id++,
      source_spec{source_spec::category::MULTI, source_spec::router::OSRM},
      station_idx.stations_, osrm_path);

  r.strategies_->net_rail_ = load_osm_net(source_spec::category::RAIL);
  r.strategies_->net_sub_ = load_osm_net(source_spec::category::SUBWAY);
  r.strategies_->net_tram_ = load_osm_net(source_spec::category::TRAM);

  r.strategies_->rel_rail_ = load_osm_rel(source_spec::category::RAIL);
  r.strategies_->rel_bus_ = load_osm_rel(source_spec::category::BUS);
  r.strategies_->rel_sub_ = load_osm_rel(source_spec::category::SUBWAY);
  r.strategies_->rel_tram_ = load_osm_rel(source_spec::category::TRAM);

  r.strategies_->rail_strategies_.push_back(r.strategies_->rel_rail_.get());
  r.strategies_->rail_strategies_.push_back(r.strategies_->net_rail_.get());

  r.strategies_->bus_strategies_.push_back(r.strategies_->rel_bus_.get());
  r.strategies_->bus_strategies_.push_back(r.strategies_->osrm_.get());

  r.strategies_->subway_strategies_.push_back(r.strategies_->rel_sub_.get());
  r.strategies_->subway_strategies_.push_back(r.strategies_->net_sub_.get());

  r.strategies_->tram_strategies_.push_back(r.strategies_->rel_tram_.get());
  r.strategies_->tram_strategies_.push_back(r.strategies_->net_tram_.get());
  r.strategies_->tram_strategies_.push_back(r.strategies_->osrm_.get());

  return r;
}

}  // namespace motis::path
