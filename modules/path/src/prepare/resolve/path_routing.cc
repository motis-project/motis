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
  std::unique_ptr<osrm_strategy> osrm_strategy_;

  std::unique_ptr<osm_strategy> network_rail_strategy_;
  std::unique_ptr<osm_strategy> network_sub_strategy_;
  std::unique_ptr<osm_strategy> network_tram_strategy_;

  std::unique_ptr<osm_strategy> relation_rail_strategy_;
  std::unique_ptr<osm_strategy> relation_bus_strategy_;
  std::unique_ptr<osm_strategy> relation_sub_strategy_;
  std::unique_ptr<osm_strategy> relation_tram_stragegy_;

  std::unique_ptr<stub_strategy> stub_strategy_;

  std::vector<routing_strategy*> rail_strategies_;
  std::vector<routing_strategy*> bus_strategies_;
  std::vector<routing_strategy*> subway_strategies_;
  std::vector<routing_strategy*> tram_strategies_;
};

path_routing::path_routing()
    : strategies_(std::make_unique<path_routing::strategies>()) {}
path_routing::~path_routing() = default;

path_routing::path_routing(path_routing&&) noexcept = default;
path_routing& path_routing::operator=(path_routing&&) noexcept = default;

std::vector<routing_strategy*> const& path_routing::strategies_for(
    source_spec::category const cat) const {

  switch (cat) {
    case source_spec::category::RAILWAY: return strategies_->rail_strategies_;
    case source_spec::category::BUS: return strategies_->bus_strategies_;
    case source_spec::category::SUBWAY: return strategies_->subway_strategies_;
    case source_spec::category::TRAM: return strategies_->tram_strategies_;
    default: throw utl::fail("unknown category");
  }
}

routing_strategy* path_routing::get_stub_strategy() const {
  return strategies_->stub_strategy_.get();
}

std::string relation_cache_name(source_spec::category const& category) {
  switch (category) {
    case source_spec::category::RAILWAY: return "pathcache.rail.relation.raw";
    case source_spec::category::SUBWAY: return "pathcache.subway.relation.raw";
    case source_spec::category::BUS: return "pathcache.bus.relation.raw";
    case source_spec::category::TRAM: return "pathcache.tram.relation.raw";
    default: throw utl::fail("relation_cache: unknown category!");
  }
}

std::string network_cache_name(source_spec::category const& category) {
  switch (category) {
    case source_spec::category::RAILWAY: return "pathcache.rail.network.raw";
    case source_spec::category::SUBWAY: return "pathcache.subway.network.raw";
    case source_spec::category::TRAM: return "pathcache.tram.network.raw";
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
  auto const load_relation_strategy = [&](std::string label,
                                          strategy_id_t const id,
                                          source_spec::category const& cat) {
    auto const fname = relation_cache_name(cat);
    auto const components =
        load_or_parse(fname, [&] { return parse_relations(osm_path, cat); });

    ml::scoped_timer t{std::string{"make osm_strategy REL "}.append(fname)};
    return std::make_unique<osm_strategy>(std::move(label), id, station_idx,
                                          components,
                                          source_spec::type::RELATION);
  };
  auto const load_network_strategy = [&](std::string label,
                                         strategy_id_t const id,
                                         source_spec::category const& cat) {
    auto const fname = network_cache_name(cat);
    auto const components =
        load_or_parse(fname, [&] { return parse_network(osm_path, cat); });

    ml::scoped_timer t{std::string{"make osm_strategy NET "}.append(fname)};
    return std::make_unique<osm_strategy>(std::move(label), id, station_idx,
                                          components,
                                          source_spec::type::RAIL_ROUTE);
  };

  path_routing r;

  strategy_id_t id = 0;

  r.strategies_->stub_strategy_ =
      std::make_unique<stub_strategy>("stub", id++, station_idx.stations_);

  r.strategies_->osrm_strategy_ = std::make_unique<osrm_strategy>(
      "osrm", id++, station_idx.stations_, osrm_path);

  r.strategies_->network_rail_strategy_ =
      load_network_strategy("net/rail", id++, source_spec::category::RAILWAY);
  r.strategies_->network_sub_strategy_ =
      load_network_strategy("net/sub ", id++, source_spec::category::SUBWAY);
  r.strategies_->network_tram_strategy_ =
      load_network_strategy("net/tram", id++, source_spec::category::TRAM);

  r.strategies_->relation_rail_strategy_ =
      load_relation_strategy("rel/rail", id++, source_spec::category::RAILWAY);
  r.strategies_->relation_bus_strategy_ =
      load_relation_strategy("rel/bus ", id++, source_spec::category::BUS);
  r.strategies_->relation_sub_strategy_ =
      load_relation_strategy("rel/sub ", id++, source_spec::category::SUBWAY);
  r.strategies_->relation_tram_stragegy_ =
      load_relation_strategy("rel/tram", id++, source_spec::category::TRAM);

  r.strategies_->rail_strategies_.push_back(
      r.strategies_->relation_rail_strategy_.get());
  r.strategies_->rail_strategies_.push_back(
      r.strategies_->network_rail_strategy_.get());

  r.strategies_->bus_strategies_.push_back(
      r.strategies_->relation_bus_strategy_.get());
  r.strategies_->bus_strategies_.push_back(r.strategies_->osrm_strategy_.get());

  r.strategies_->subway_strategies_.push_back(
      r.strategies_->relation_sub_strategy_.get());
  r.strategies_->subway_strategies_.push_back(
      r.strategies_->network_sub_strategy_.get());

  r.strategies_->tram_strategies_.push_back(
      r.strategies_->relation_tram_stragegy_.get());
  r.strategies_->tram_strategies_.push_back(
      r.strategies_->network_tram_strategy_.get());
  r.strategies_->tram_strategies_.push_back(
      r.strategies_->osrm_strategy_.get());

  return r;
}

}  // namespace motis::path
