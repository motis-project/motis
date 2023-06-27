#include <utility>

#include "motis/footpaths/platforms.h"

#include "boost/algorithm/string.hpp"
#include "motis/core/common/logging.h"
#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/geom/coordinates.hpp"
#include "osmium/handler/node_locations_for_ways.hpp"
#include "osmium/index/map/flex_mem.hpp"
#include "osmium/visitor.hpp"

#include "utl/pipes.h"

namespace logging = motis::logging;

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
                            osmium::TagsFilter filter)
      : platforms_(platforms), filter_(std::move(filter)){};

  void node(osmium::Node const& node) {
    auto const& tags = node.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_track(osm_type::NODE, node.id(), node.location(), tags);
    }
  }

  void way(osmium::Way const& way) {
    auto const& tags = way.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_track(osm_type::WAY, way.id(), way.envelope().bottom_left(), tags);
    }
  }

  void area(osmium::Area const& area) {
    auto const& tags = area.tags();
    if (osmium::tags::match_any_of(tags, filter_)) {
      add_track(area.from_way() ? osm_type::WAY : osm_type::RELATION,
                area.orig_id(), calc_center(*area.cbegin<osmium::OuterRing>()),
                tags);
    }
  }

private:
  void add_track(osm_type const type, osmium::object_id_type const id,
                 osmium::geom::Coordinates const& coord,
                 osmium::TagList const& tags) {
    auto names = extract_platform_names(tags);
    platforms_.emplace_back(names.front(), id, type,
                            geo::latlng{coord.y, coord.x});

    /**
    // TODO (Carsten) Update Postprocess names; Names are used in lookup;
    // TODO (Carsten)
    auto const name = names.back();

    names.clear();
    boost::split(names, name, [](char c) { return c == ';' || c == '/'; });
    bool use_only_one_name =
        std::any_of(names.begin(), names.end(),
                    [](std::string const& name) { return name.length() > 3; });

    if (use_only_one_name) {
      platforms_.emplace_back(
          track_info{id, type, names.front(), geo::latlng{coord.y, coord.x}});
    } else {
      for (auto const& name : names) {
        platforms_.emplace_back(
            track_info{id, type, name, geo::latlng{coord.y, coord.x}});
      }
    }
     */
  }

  std::vector<platform_info>& platforms_;
  osmium::TagsFilter filter_;
};

std::vector<platform_info> extract_osm_platforms(std::string const& osm_file) {

  ::logging::scoped_timer const timer("Extract OSM Tracks from " + osm_file);

  osmium::io::File const input_file{osm_file};

  osmium::area::Assembler::config_type assembler_config;
  assembler_config.create_empty_areas = false;
  osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{
      assembler_config};

  osmium::TagsFilter filter{false};
  filter.add_rule(true, "public_transport", "platform");
  filter.add_rule(true, "railway", "platform");

  std::clog << "Extract OSM Tracks: Pass 1..." << std::endl;
  osmium::relations::read_relations(input_file, mp_manager);

  index_type index;
  location_handler_type location_handler{index};
  std::vector<platform_info> platforms;
  platform_handler data_handler{platforms, filter};

  std::clog << "Extract OSM Tracks: Pass 2...";
  osmium::io::Reader reader{input_file, osmium::io::read_meta::no};
  osmium::apply(reader, location_handler, data_handler,
                mp_manager.handler(
                    [&data_handler](const osmium::memory::Buffer& area_buffer) {
                      osmium::apply(area_buffer, data_handler);
                    }));

  reader.close();

  std::clog << "Extracted " << platforms.size() << "platforms from OSM."
            << std::endl;

  return platforms;
}

std::vector<std::string> extract_platform_names(osmium::TagList const& tags) {
  std::vector<std::string> platform_names;
  if (tags.has_key("name")) {
    platform_names.emplace_back(tags["name"]);
  }
  if (tags.has_key("ref_name")) {
    platform_names.emplace_back(tags["ref_name"]);
  }
  if (tags.has_key("local_ref")) {
    platform_names.emplace_back(tags["local_ref"]);
  }
  if (tags.has_key("ref")) {
    platform_names.emplace_back(tags["ref"]);
  }
  return platform_names;
}

std::vector<platform_info> platforms::get_platforms_in_radius(
    geo::latlng const& loc, double radius) {
  return utl::all(platform_index_.in_radius(loc, radius)) |
         utl::transform([this](std::size_t i) { return platforms_[i]; }) |
         utl::vec();
}

}  // namespace motis::footpaths
