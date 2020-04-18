#pragma once

#include <set>
#include <string>
#include <vector>

namespace motis::path {

struct source_spec {
  enum class category : uint8_t { UNKNOWN, MULTI, BUS, TRAM, SUBWAY, RAIL };
  enum class router : uint8_t { STUB, OSRM, OSM_NET, OSM_REL };

  source_spec() = default;
  source_spec(category c, router r) : category_(c), router_(r) {}

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

  category category_;
  router router_;
};

template <typename Fun>
void foreach_path_category(std::set<int> const& motis_categories, Fun&& fun) {
  std::vector<uint32_t> railway_cat, unknown_cat;
  for (auto const& category : motis_categories) {
    if (category == 8) {
      fun(source_spec::category::BUS, std::vector<uint32_t>{8});
    } else if (category == 7) {
      fun(source_spec::category::TRAM, std::vector<uint32_t>{7});
    } else if (category == 6) {
      fun(source_spec::category::SUBWAY, std::vector<uint32_t>{6});
    } else if (category < 6) {
      railway_cat.push_back(category);
    } else {
      unknown_cat.push_back(category);
    }
  }

  if (!railway_cat.empty()) {
    fun(source_spec::category::RAIL, railway_cat);
  }
  if (!unknown_cat.empty()) {
    fun(source_spec::category::UNKNOWN, unknown_cat);
  }
}

}  // namespace motis::path
