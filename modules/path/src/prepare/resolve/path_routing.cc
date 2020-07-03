#include "motis/path/prepare/resolve/path_routing.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/strategy/osm_strategy.h"
#include "motis/path/prepare/strategy/osrm_strategy.h"
#include "motis/path/prepare/strategy/stub_strategy.h"

namespace motis::path {

struct path_routing::strategies {
  std::unique_ptr<stub_strategy> stub_;
  std::unique_ptr<osrm_strategy> osrm_;
  std::unique_ptr<osm_strategy> net_rail_, net_sub_, net_tram_, net_ship_;
  std::unique_ptr<osm_strategy> rel_rail_, rel_sub_, rel_tram_, rel_bus_;

  std::vector<routing_strategy*> rail_strategies_, tram_strategies_,
      subway_strategies_, bus_strategies_, ship_strategies_;
};

path_routing::path_routing()
    : strategies_{std::make_unique<path_routing::strategies>()} {}
path_routing::~path_routing() = default;

std::vector<routing_strategy*> const& path_routing::strategies_for(
    source_spec::category const cat) const {
  switch (cat) {
    case source_spec::category::BUS: return strategies_->bus_strategies_;
    case source_spec::category::TRAM: return strategies_->tram_strategies_;
    case source_spec::category::SUBWAY: return strategies_->subway_strategies_;
    case source_spec::category::RAIL: return strategies_->rail_strategies_;
    case source_spec::category::SHIP: return strategies_->ship_strategies_;
    default: throw utl::fail("no strategies for this category");
  }
}

routing_strategy* path_routing::get_stub_strategy() const {
  return strategies_->stub_.get();
}

path_routing make_path_routing(station_index const& station_idx,
                               osm_data const& osm_data,
                               std::string const& osrm_path) {
  using category = source_spec::category;
  using router = source_spec::router;

  path_routing r;
  auto& s = *r.strategies_;

  strategy_id_t id = 0ULL;
  s.stub_ = std::make_unique<stub_strategy>(
      id++, source_spec{category::MULTI, router::STUB}, station_idx.stations_);

  s.osrm_ = std::make_unique<osrm_strategy>(
      id++, source_spec{category::MULTI, router::OSRM}, station_idx.stations_,
      osrm_path);

  auto const make_osm_strategy = [&](source_spec const ss) {
    return std::make_unique<osm_strategy>(id++, ss, station_idx,
                                          osm_data.profiles_.at(ss));
  };

  s.net_rail_ = make_osm_strategy({category::RAIL, router::OSM_NET});
  s.net_sub_ = make_osm_strategy({category::SUBWAY, router::OSM_NET});
  s.net_tram_ = make_osm_strategy({category::TRAM, router::OSM_NET});
  s.net_ship_ = make_osm_strategy({category::SHIP, router::OSM_NET});

  s.rel_rail_ = make_osm_strategy({category::RAIL, router::OSM_REL});
  s.rel_sub_ = make_osm_strategy({category::SUBWAY, router::OSM_REL});
  s.rel_tram_ = make_osm_strategy({category::TRAM, router::OSM_REL});
  s.rel_bus_ = make_osm_strategy({category::BUS, router::OSM_REL});

  s.bus_strategies_ = {s.rel_bus_.get(), s.osrm_.get()};
  s.subway_strategies_ = {s.rel_sub_.get(), s.net_sub_.get()};
  s.tram_strategies_ = {s.rel_tram_.get(), s.net_tram_.get(), s.osrm_.get()};
  s.rail_strategies_ = {s.rel_rail_.get(), s.net_rail_.get()};
  s.ship_strategies_ = {s.net_ship_.get()};

  return r;
}

}  // namespace motis::path
