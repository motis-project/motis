#pragma once

#include <set>
#include <string>
#include <vector>

namespace motis::path {

struct source_spec {
  enum class category { UNKNOWN, RAILWAY, SUBWAY, BUS, TRAM };

  enum class type { RELATION, OSRM_ROUTE, STUB_ROUTE, RAIL_ROUTE };

  source_spec() = default;
  source_spec(int64_t id, category c, type t)
      : id_(id), category_(c), type_(t) {}

  std::string type_str() const {
    switch (type_) {
      case type::RELATION: return "RELATION";
      case type::OSRM_ROUTE: return "OSRM_ROUTE";
      case type::STUB_ROUTE: return "STUB_ROUTE";
      case type::RAIL_ROUTE: return "RAIL_ROUTE";
      default: return "INVALID";
    }
  }

  int64_t id_;
  category category_;
  type type_;
};

inline bool operator==(source_spec::category const t, int const category) {
  switch (t) {
    case source_spec::category::RAILWAY: return category < 6;
    default: return category >= 6;
  };
}

template <typename Fun>
void foreach_path_category(std::set<int> const& motis_categories, Fun&& fun) {
  std::vector<uint32_t> railway_cat;
  std::vector<uint32_t> bus_cat;
  std::vector<uint32_t> subway_cat;
  std::vector<uint32_t> tram_cat;
  std::vector<uint32_t> unknown_cat;

  for (auto const& category : motis_categories) {
    if (category < 6) {
      railway_cat.push_back(category);
    } else if (category == 8) {
      bus_cat.push_back(category);
    } else if (category == 6) {
      subway_cat.push_back(category);
    } else if (category == 7) {
      tram_cat.push_back(category);
    } else {
      unknown_cat.push_back(category);
    }
  }

  if (!railway_cat.empty()) {
    fun(source_spec::category::RAILWAY, railway_cat);
  }
  if (!bus_cat.empty()) {
    fun(source_spec::category::BUS, bus_cat);
  }
  if (!subway_cat.empty()) {
    fun(source_spec::category::SUBWAY, subway_cat);
  }
  if (!tram_cat.empty()) {
    fun(source_spec::category::TRAM, tram_cat);
  }
  if (!unknown_cat.empty()) {
    fun(source_spec::category::UNKNOWN, unknown_cat);
  }
}

}  // namespace motis::path
