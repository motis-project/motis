#include <fstream>
#include <iostream>
#include <string_view>

#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/geom/coordinates.hpp"
#include "osmium/handler/node_locations_for_ways.hpp"
#include "osmium/index/map/flex_mem.hpp"
#include "osmium/io/pbf_input.hpp"
#include "osmium/visitor.hpp"

#include "motis/core/common/logging.h"

#include "motis/parking/osm_parking_lots.h"

using namespace motis::logging;

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type,
                                               osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

namespace motis::parking {

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

class parking_handler : public osmium::handler::Handler {
public:
  explicit parking_handler(std::vector<parking_lot>& parking_lots)
      : parking_lots_{parking_lots} {}

  void node(osmium::Node const& node) {
    auto const& tags = node.tags();
    if (tags.has_tag("amenity", "parking")) {
      add_parking(osm_type::NODE, node.id(), node.location(), tags);
    }
  }

  void area(osmium::Area const& area) {
    auto const& tags = area.tags();
    if (tags.has_tag("amenity", "parking")) {
      add_parking(area.from_way() ? osm_type::WAY : osm_type::RELATION,
                  area.orig_id(),
                  calc_center(*area.cbegin<osmium::OuterRing>()), tags);
    }
  }

private:
  void add_parking(osm_type const ot, osmium::object_id_type const id,
                   osmium::geom::Coordinates const& coord,
                   osmium::TagList const& tags) {
    if (!access_allowed(tags)) {
      return;
    }
    parking_lots_.emplace_back(parking_lot{
        0, geo::latlng{coord.y, coord.x},
        parking_lot::info_t{osm_parking_lot_info{id, ot, get_fee_type(tags)}}});
  }

  static inline fee_type get_fee_type(osmium::TagList const& tags) {
    using namespace std::literals;
    if (tags.has_key("fee")) {
      auto const val = tags["fee"];
      if (val == "yes"sv) {
        return fee_type::YES;
      } else if (val == "no"sv) {
        return fee_type::NO;
      }
    }
    return fee_type::UNKNOWN;
  }

  static inline bool access_allowed(osmium::TagList const& tags) {
    auto const access = tags["access"];
    if (access == nullptr) {
      return true;
    }
    return strcmp(access, "yes") == 0 || strcmp(access, "permissive") == 0 ||
           strcmp(access, "public") == 0;
  }

  std::vector<parking_lot>& parking_lots_;
};

std::vector<parking_lot> extract_osm_parking_lots(std::string const& osm_file) {
  scoped_timer const timer("Extracting OSM parking lot data");

  osmium::io::File const input_file{osm_file};

  osmium::area::Assembler::config_type assembler_config;
  assembler_config.create_empty_areas = false;
  osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{
      assembler_config};

  std::clog << "Extract OSM parking lots: Pass 1..." << '\n';
  osmium::relations::read_relations(input_file, mp_manager);

  index_type index;
  location_handler_type location_handler{index};
  std::vector<parking_lot> parking_lots;
  parking_handler data_handler{parking_lots};

  std::clog << "Extract OSM parking lots: Pass 2..." << '\n';
  osmium::io::Reader reader{input_file, osmium::io::read_meta::no};
  osmium::apply(reader, location_handler, data_handler,
                mp_manager.handler(
                    [&data_handler](const osmium::memory::Buffer& area_buffer) {
                      osmium::apply(area_buffer, data_handler);
                    }));

  reader.close();

  std::clog << "Extracted " << parking_lots.size() << " parking lots from OSM"
            << '\n';

  return parking_lots;
}

}  // namespace motis::parking
