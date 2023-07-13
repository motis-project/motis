#include <utility>

#include "boost/algorithm/string.hpp"

#include "motis/core/common/logging.h"

#include "motis/footpaths/platforms.h"

#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/geom/coordinates.hpp"
#include "osmium/handler/node_locations_for_ways.hpp"
#include "osmium/index/map/flex_mem.hpp"
#include "osmium/visitor.hpp"

#include "utl/pipes.h"

using namespace motis::logging;

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type,
                                               osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

namespace motis::footpaths {

osmium::geom::Coordinates calc_center(osmium::NodeRefList const& nr_list) {
  osmium::geom::Coordinates c{0.0, 0.0};

  for (auto const& nr : nr_list) {
    c.x += nr.lon();
    c.y += nr.lat();
  }

  c.x /= nr_list.size();
  c.y /= nr_list.size();

  return c;
}

struct platform_handler : public osmium::handler::Handler {
  explicit platform_handler(std::vector<platform_info>& platforms,
                            osmium::TagsFilter const& filter)
      : platforms_(platforms), filter_(filter){};

  void node(osmium::Node const& node) {
    auto const& tags = node.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_platform(nigiri::osm_type::NODE, node.id(), node.location(), tags);
    }
  }

  void way(osmium::Way const& way) {
    auto const& tags = way.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_platform(nigiri::osm_type::WAY, way.id(),
                   way.envelope().bottom_left(), tags);
    }
  }

  void area(osmium::Area const& area) {
    auto const& tags = area.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_platform(
          area.from_way() ? nigiri::osm_type::WAY : nigiri::osm_type::RELATION,
          area.orig_id(), calc_center(*area.cbegin<osmium::OuterRing>()), tags);
    }
  }

  u_int unique_platforms_{0};

private:
  void add_platform(nigiri::osm_type const type,
                    osmium::object_id_type const id,
                    osmium::geom::Coordinates const& coord,
                    osmium::TagList const& tags) {
    // TODO (Carsten) insert "more"/all names
    auto names = extract_platform_names(tags);

    if (!names.empty()) {
      ++unique_platforms_;
    }

    for (auto const& name : names) {
      platforms_.emplace_back(name, id, type, geo::latlng{coord.y, coord.x},
                              platform_is_bus_stop(tags));
    }
  }

  std::vector<platform_info>& platforms_;
  osmium::TagsFilter filter_;
};

std::vector<platform_info> extract_osm_platforms(std::string const& osm_file) {

  scoped_timer const timer("Extract OSM Tracks from " + osm_file);

  osmium::io::File const input_file{osm_file};

  osmium::area::Assembler::config_type assembler_config;
  assembler_config.create_empty_areas = false;
  osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{
      assembler_config};

  osmium::TagsFilter filter{false};
  filter.add_rule(true, "public_transport", "platform");
  filter.add_rule(true, "railway", "platform");

  {
    scoped_timer const timer("Extract OSM tracks: Pass 1...");
    osmium::relations::read_relations(input_file, mp_manager);
  }

  index_type index;
  location_handler_type location_handler{index};
  std::vector<platform_info> platforms;
  platform_handler data_handler{platforms, filter};

  {
    scoped_timer const timer("Extract OSM tracks: Pass 2...");

    osmium::io::Reader reader{input_file, osmium::io::read_meta::no};
    osmium::apply(
        reader, location_handler, data_handler,
        mp_manager.handler(
            [&data_handler](const osmium::memory::Buffer& area_buffer) {
              osmium::apply(area_buffer, data_handler);
            }));

    reader.close();
  }

  LOG(info) << "Extracted " << data_handler.unique_platforms_
            << " unique platforms from OSM.";
  LOG(info) << "Generated " << platforms.size() << " platform_info structs. "
            << static_cast<float>(platforms.size()) /
                   static_cast<float>(data_handler.unique_platforms_)
            << " entries per platform.";

  return platforms;
}

std::vector<std::string> extract_platform_names(osmium::TagList const& tags) {
  std::vector<std::string> platform_names;

  auto add_names = [&](std::string name_by_tag) {
    platform_names.emplace_back(name_by_tag);
    return;
    /**
     * // In case matching is invalid: try to split names; currently not needed
     *
     * std::vector<std::string> names{};
     * boost::split(names, name_by_tag,
     *              [](char c) { return c == ';' || c == '/'; });
     *
     * if (std::any_of(names.begin(), names.end(),
     *   [&](std::string const& name) -> bool {
     *     return name.length() > 3;
     *    })) {
     *       platform_names.emplace_back(name_by_tag);
     *       return;
     *     }
     *
     * for (auto const& name : names) {
     *   platform_names.emplace_back(name);
     * }
     */
  };

  // REMOVE *.clear() to get more names for matching
  // TODO (Carsten) find a better way of matching
  if (tags.has_key("name")) {
    platform_names.clear();
    add_names(tags.get_value_by_key("name"));
  }
  if (tags.has_key("description")) {
    platform_names.clear();
    add_names(tags.get_value_by_key("description"));
  }
  if (tags.has_key("ref_name")) {
    platform_names.clear();
    add_names(tags.get_value_by_key("ref_name"));
  }
  if (tags.has_key("local_ref")) {
    platform_names.clear();
    add_names(tags.get_value_by_key("local_ref"));
  }
  if (tags.has_key("ref")) {
    platform_names.clear();
    add_names(tags.get_value_by_key("ref"));
  }

  return platform_names;
}

bool platform_is_bus_stop(osmium::TagList const& tags) {
  return (tags.has_key("highway") &&
          strcmp(tags.get_value_by_key("highway"), "bus_stop") == 0);
}

std::vector<platform_info*> platforms::get_valid_platforms_in_radius(
    platform_info* platform, double radius) {
  return utl::all(platform_index_.in_radius(platform->pos_, radius)) |
         utl::transform(
             [this](std::size_t i) { return get_platform_info(i); }) |
         utl::remove_if([&](auto* target_platform) {
           return target_platform->idx_ == nigiri::location_idx_t::invalid() ||
                  target_platform->idx_ == platform->idx_;
         }) |
         utl::vec();
}

std::vector<platform_info*> platforms::get_platforms_in_radius(
    geo::latlng loc, double const radius) {
  return utl::all(platform_index_.in_radius(loc, radius)) |
         utl::transform(
             [this](std::size_t i) { return get_platform_info(i); }) |
         utl::vec();
}

}  // namespace motis::footpaths
