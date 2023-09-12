#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "motis/transfers/platform/platform.h"
#include "motis/transfers/types.h"

#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/geom/coordinates.hpp"
#include "osmium/handler.hpp"
#include "osmium/io/file.hpp"
#include "osmium/osm/area.hpp"
#include "osmium/osm/node.hpp"
#include "osmium/osm/node_ref_list.hpp"
#include "osmium/osm/tag.hpp"
#include "osmium/osm/types.hpp"
#include "osmium/osm/way.hpp"
#include "osmium/relations/manager_util.hpp"

namespace motis::transfers {

struct osm_platform_extractor {
  explicit osm_platform_extractor(std::filesystem::path const& osm_file_path)
      : osm_file_{osm_file_path.c_str()} {
    osmium::relations::read_relations(osm_file_, mp_manager_);
  }

  // extracts platforms according to the set filter rules and name tags from the
  // `osm_file_`.
  // Requirements: at least one filter rule and one name tag have
  // been set.
  // - `!filter_.empty() && !osm_name_tag_keys_.empty()` evaluates to true.
  platforms get_platforms_identified_in_osm_file();

  // Adds a rule according to which the platforms should be extracted from the
  // OSM file.
  void add_filter_rule(bool const, std::string const&, std::string const&);

  // Adds a key that will be searched for in the tag list for a name.
  void add_platform_name_tag_key(std::string const&);

private:
  // Returns the center coordinate of a list of osm node references by
  // calculating the mean of the longitude and latitude of all given coords.
  // - x = mean(sum(nodes.lon);
  // - y = mean(sum(nodes.lat));
  static osmium::geom::Coordinates calc_center(osmium::NodeRefList const&);

  struct platform_handler : public osmium::handler::Handler {
    platforms platforms_;
    osmium::TagsFilter filter_{false};
    std::vector<std::string> osm_name_tag_keys_;

    // Handler command for handling platforms described as osm nodes.
    void node(osmium::Node const&);

    // Handler command for handling platforms described as osm ways.
    void way(osmium::Way const&);

    // Handler command for handling platforms described as osm areas.
    void area(osmium::Area const&);

  private:
    // Adds a new platform with the given information to the list of extracted
    // platforms.
    // Equivalent to: platforms_.emplace_back(platform{...})
    void add_platform(osm_type const, osmium::object_id_type const,
                      osmium::geom::Coordinates const&,
                      strings const& /* names */, bool /* is_bus_stop */);

    // Checks wether the given tag lists describes a platform or not.
    // Equivalent to: osmium::tags::match_any_of(tags, filter);
    bool is_platform(osmium::TagList const&) const;

    // Determines whether a given osm platform described with a tag list is a
    // bus stop or not.
    static bool platform_is_bus_stop(osmium::TagList const&);

    // Returns a list of known and unique osm names of the given osm platform
    // described with a tag list.
    strings get_platform_names(osmium::TagList const&);
  } platform_handler_;

  osmium::io::File osm_file_;
  osmium::area::Assembler::config_type assembler_config_;
  osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager_{
      assembler_config_};
};

}  // namespace motis::transfers