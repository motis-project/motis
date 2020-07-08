#include "motis/path/prepare/resolve/path_routing.h"

#include "boost/filesystem.hpp"

#include "utl/enumerate.h"

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

path_routing make_path_routing(mcd::vector<station_seq> const& sequences,
                               station_index const& station_idx,
                               osm_data const& osm_data,
                               std::string const& osrm_path) {
  using category = source_spec::category;
  using router = source_spec::router;

  std::array<bool, static_cast<service_class_t>(service_class::NUM_CLASSES)>
      active_classes_acc{};
  active_classes_acc.fill(false);
  for (auto const& seq : sequences) {
    for (auto const clasz : seq.classes_) {
      active_classes_acc[static_cast<service_class_t>(clasz)] = true;
    }
  }
  std::vector<service_class> active_classes;
  for (auto const& [clasz, active] : utl::enumerate(active_classes_acc)) {
    if (active) {
      active_classes.push_back(static_cast<service_class>(clasz));
    }
  }

  path_routing r;
  auto& s = *r.strategies_;
  strategy_id_t id = 0ULL;
  s.stub_ = std::make_unique<stub_strategy>(
      id++, source_spec{category::MULTI, router::STUB}, station_idx.stations_);

  auto const init_osrm = [&](std::unique_ptr<osrm_strategy>& ptr) {
    if (ptr == nullptr) {
      LOG(ml::info) << id << " : load osrm_strategy";
      ptr = std::make_unique<osrm_strategy>(
          id++, source_spec{category::MULTI, router::OSRM},
          station_idx.stations_, osrm_path);
    }
    return ptr.get();
  };

  auto const init_osm2 = [&](std::unique_ptr<osm_strategy>& ptr,
                             source_spec const input, source_spec const graph) {
    if (ptr == nullptr) {
      LOG(ml::info) << id << " : load osm_strategy " << graph.str();
      ptr = std::make_unique<osm_strategy>(id++, graph, station_idx,
                                           osm_data.profiles_.at(input));
    }
    return ptr.get();
  };
  auto const init_osm = [&](std::unique_ptr<osm_strategy>& ptr,
                            source_spec const ss) {
    return init_osm2(ptr, ss, ss);
  };

  foreach_path_category(active_classes, [&](auto const cat, auto const&) {
    switch (cat) {
      case source_spec::category::BUS:
        s.bus_strategies_ = {
            init_osm(s.rel_bus_, {category::BUS, router::OSM_REL}),
            init_osrm(s.osrm_)};
        break;
      case source_spec::category::TRAM:
        s.tram_strategies_ = {
            init_osm(s.rel_tram_, {category::TRAM, router::OSM_REL}),
            init_osm2(s.net_tram_, {category::MULTI, router::OSM_NET},
                      {category::TRAM, router::OSM_NET}),
            init_osrm(s.osrm_)};
        break;
      case source_spec::category::SUBWAY:
        s.subway_strategies_ = {
            init_osm(s.rel_sub_, {category::SUBWAY, router::OSM_REL}),
            init_osm2(s.net_sub_, {category::MULTI, router::OSM_NET},
                      {category::SUBWAY, router::OSM_NET})};
        break;
      case source_spec::category::RAIL:
        s.rail_strategies_ = {
            init_osm(s.rel_rail_, {category::RAIL, router::OSM_REL}),
            init_osm2(s.net_rail_, {category::MULTI, router::OSM_NET},
                      {category::RAIL, router::OSM_NET})};
        break;
      case source_spec::category::SHIP:
        s.ship_strategies_ = {init_osm2(s.net_ship_,
                                        {category::MULTI, router::OSM_NET},
                                        {category::SHIP, router::OSM_NET})};
        break;
      case source_spec::category::UNKNOWN: break;
      default: throw utl::fail("no strategies for this category");
    }
  });

  return r;
}

}  // namespace motis::path
