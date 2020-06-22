#pragma once

#include <set>
#include <string>
#include <vector>

#include "cista/hash.h"
#include "cista/reflection/comparable.h"

#include "motis/path/definitions.h"

namespace motis::path {

struct source_spec {
  CISTA_COMPARABLE();

  enum class category : uint8_t { UNKNOWN, MULTI, BUS, TRAM, SUBWAY, RAIL };
  enum class router : uint8_t { STUB, OSRM, OSM_NET, OSM_REL };

  std::string category_str() const {
    switch (category_) {
      case category::UNKNOWN: return "UNKNOWN";  // MOTIS "class 9"
      case category::MULTI: return "MULTI";  // PATH: stub/osrm strategy
      case category::BUS: return "BUS";
      case category::TRAM: return "TRAM";
      case category::SUBWAY: return "SUBWAY";
      case category::RAIL: return "RAIL";
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
void foreach_path_category(std::set<motis_clasz_t> const& classes, Fun&& fun) {
  std::vector<motis_clasz_t> railway_cat, unknown_cat;
  for (auto const& clasz : classes) {
    if (clasz == 8) {
      fun(source_spec::category::BUS, std::vector<motis_clasz_t>{8});
    } else if (clasz == 7) {
      fun(source_spec::category::TRAM, std::vector<motis_clasz_t>{7});
    } else if (clasz == 6) {
      fun(source_spec::category::SUBWAY, std::vector<motis_clasz_t>{6});
    } else if (clasz < 6) {
      railway_cat.push_back(clasz);
    } else {
      unknown_cat.push_back(clasz);
    }
  }

  if (classes.empty()) {
    unknown_cat.push_back(9U);
  }

  if (!railway_cat.empty()) {
    fun(source_spec::category::RAIL, railway_cat);
  }
  if (!unknown_cat.empty()) {
    fun(source_spec::category::UNKNOWN, unknown_cat);
  }
}

}  // namespace motis::path
