#pragma once

#include <set>
#include <string>
#include <vector>

#include "cista/hash.h"
#include "cista/reflection/comparable.h"

#include "motis/core/schedule/connection.h"

namespace motis::path {

struct source_spec {
  CISTA_COMPARABLE();

  enum class category : uint8_t {
    UNKNOWN,
    MULTI,
    BUS,
    TRAM,
    SUBWAY,
    RAIL,
    SHIP
  };
  enum class router : uint8_t { STUB, OSRM, OSM_NET, OSM_REL };

  std::string category_str() const {
    switch (category_) {
      case category::UNKNOWN: return "UNKNOWN";  // MOTIS "class 12"
      case category::MULTI: return "MULTI";  // PATH: stub/osrm strategy
      case category::BUS: return "BUS";
      case category::TRAM: return "TRAM";
      case category::SUBWAY: return "SUBWAY";
      case category::RAIL: return "RAIL";
      case category::SHIP: return "SHIP";
      default: return "INVALID";
    }
  }

  std::string router_str() const {
    switch (router_) {
      case router::STUB: return "STUB";
      case router::OSRM: return "OSRM";
      case router::OSM_NET: return "OSM_NET";
      case router::OSM_REL: return "OSM_REL";
      default: return "INVALID";
    }
  }

  std::string str() const {
    return category_str().append("/").append(router_str());
  }

  size_t hash() const {
    return cista::hash_combine(
        cista::BASE_HASH,
        static_cast<typename std::underlying_type_t<category>>(category_),
        static_cast<typename std::underlying_type_t<router>>(router_));
  }

  category category_;
  router router_;
};

template <typename Fun>
void foreach_path_category(mcd::vector<service_class> const& classes,
                           Fun&& fun) {
  mcd::vector<service_class> railway_classes, bus_classes, other_classes;
  for (auto const& clasz : classes) {
    if (clasz == service_class::SHIP) {
      fun(source_spec::category::SHIP,
          mcd::vector<service_class>{service_class::SHIP});
    } else if (clasz == service_class::BUS || clasz == service_class::COACH) {
      bus_classes.push_back(clasz);
    } else if (clasz == service_class::STR) {
      fun(source_spec::category::TRAM,
          mcd::vector<service_class>{service_class::STR});
    } else if (clasz == service_class::U) {
      fun(source_spec::category::SUBWAY,
          mcd::vector<service_class>{service_class::U});
    } else if (clasz == service_class::ICE || clasz == service_class::IC ||
               clasz == service_class::N || clasz == service_class::RE ||
               clasz == service_class::RB || clasz == service_class::S) {
      railway_classes.push_back(clasz);
    } else {
      other_classes.push_back(clasz);  // also: AIR
    }
  }

  if (classes.empty()) {
    other_classes.push_back(service_class::OTHER);
  }

  if (!railway_classes.empty()) {
    fun(source_spec::category::RAIL, railway_classes);
  }
  if (!bus_classes.empty()) {
    fun(source_spec::category::BUS, bus_classes);
  }
  if (!other_classes.empty()) {
    fun(source_spec::category::UNKNOWN, other_classes);
  }
}

}  // namespace motis::path
