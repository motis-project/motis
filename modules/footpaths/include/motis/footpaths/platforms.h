#pragma once

#include <string>
#include <utility>
#include <vector>

#include "osmium/tags/taglist.hpp"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/types.h"

namespace motis::footpaths {
enum class osm_type : std::uint8_t { NODE, WAY, RELATION };

struct platform_info {
  platform_info() = default;
  platform_info(std::string name, int64_t osm_id, osm_type osm_type,
                geo::latlng pos)
      : name_(std::move(name)),
        osm_id_(osm_id),
        osm_type_(osm_type),
        pos_(pos){};
  platform_info(std::string name, nigiri::location_idx_t idx, geo::latlng pos)
      : name_(std::move(name)), idx_(idx), pos_(pos){};

  std::string name_;
  int64_t osm_id_{-1};
  nigiri::location_idx_t idx_{nigiri::location_idx_t::invalid()};
  osm_type osm_type_{osm_type::NODE};
  geo::latlng pos_;  // used to calculate distance to other tracks
};

struct platforms {
  std::vector<platform_info> platforms_;

  explicit platforms(std::vector<platform_info> platforms)
      : platforms_(std::move(platforms)) {
    platform_index_ =
        geo::make_point_rtree(platforms_, [](auto const& p) { return p.pos_; });
  }

  /**
   * Calculates all platforms that are within a radius of the given location.
   *
   * @param loc the location in the vicinity of which platforms are searched
   * for.
   * @param radius the maximal allowed distance |location, platform.pos|   *
   */
  std::vector<platform_info> get_platforms_in_radius(geo::latlng const& loc,
                                                     double radius);

  std::size_t size() const { return platforms_.size(); }

  platform_info* get_platform_info(std::size_t const& i) {
    return &platforms_.at(i);
  };

private:
  geo::point_rtree platform_index_;
};

/**
 * Extracts all platforms in a given osm_file at least once.
 *
 * A platform can appear several times in the results list depending on the
 * given names of this platform.
 *
 * @param osm_file path/to/osm_file
 * @return a list of platforms found in osm_file
 */
std::vector<platform_info> extract_osm_platforms(std::string const& osm_file);

/**
 * Extract all OSM Object Names from a given taglist.
 *
 * @param tags a list of key-value pairs that contain at least one name
 * reference.
 * @return a list of names found in tags.
 */
std::vector<std::string> extract_platform_names(osmium::TagList const& tags);

}  // namespace motis::footpaths