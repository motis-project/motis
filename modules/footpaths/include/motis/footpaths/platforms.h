#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cista/containers/string.h"

#include "osmium/tags/taglist.hpp"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/types.h"

#include "ppr/routing/input_location.h"

namespace motis::footpaths {
enum class osm_type { kNode, kWay, kRelation, kUnknown };

inline char get_osm_str_type(osm_type const ot) {
  switch (ot) {
    case osm_type::kNode: return 'n';
    case osm_type::kWay: return 'w';
    case osm_type::kRelation: return 'r';
    default: return '_';
  }
}

struct platform_info {
  cista::raw::string_view name_;
  std::int64_t osm_id_{-1};
  nigiri::location_idx_t idx_{nigiri::location_idx_t::invalid()};
  osm_type osm_type_{osm_type::kNode};
  bool is_bus_stop_{false};
};

struct platform {
  std::int64_t id_{-1};
  geo::latlng loc_;

  platform_info info_;
};

struct platforms_index {

  explicit platforms_index(std::vector<platform>& pfs)
      : platforms_(std::move(pfs)) {
    platform_index_ = geo::make_point_rtree(
        platforms_, [](auto const& pf) { return pf.loc_; });
  }

  std::vector<platform*> valid_in_radius(platform const*, double const);

  std::vector<platform*> in_radius(geo::latlng const, double const);
  std::vector<std::pair<double, platform*>> in_radius_with_distance(
      geo::latlng const, double const);

  std::size_t size() const { return platforms_.size(); }

  platform* get_platform(std::size_t const i) { return &platforms_.at(i); };

  std::vector<platform> platforms_;

private:
  geo::point_rtree platform_index_;
};

std::vector<platform> extract_osm_platforms(std::string const& osm_file);

std::vector<std::string> extract_platform_names(osmium::TagList const& tags);

bool platform_is_bus_stop(osmium::TagList const& tags);

/**
 * Returns the equivalent OSM_TYPE of PPR.
 * Default: ::ppr::routing::osm_namespace::NODE
 */
::ppr::routing::osm_namespace to_ppr_osm_type(osm_type const&);

::ppr::routing::input_location to_input_location(platform const&);

}  // namespace motis::footpaths