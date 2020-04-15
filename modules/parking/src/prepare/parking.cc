#include <fstream>
#include <iostream>

#include "boost/filesystem.hpp"

#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/geom/coordinates.hpp"
#include "osmium/handler/node_locations_for_ways.hpp"
#include "osmium/index/map/flex_mem.hpp"
#include "osmium/io/pbf_input.hpp"
#include "osmium/visitor.hpp"

#include "motis/core/common/logging.h"
#include "motis/parking/prepare/parking.h"

namespace fs = boost::filesystem;
using namespace motis::logging;

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type,
                                               osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

namespace motis::parking::prepare {

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
  explicit parking_handler(std::string const& parking_file,
                           std::vector<parking_lot>& parkings)
      : out_(parking_file), parkings_(parkings) {
    out_.precision(12);
    out_ << "lat,lng,fee\n";
  }

  void node(osmium::Node const& node) {
    auto const& tags = node.tags();
    if (tags.has_tag("amenity", "parking")) {
      add_parking(node.location(), tags);
    }
  }

  void area(osmium::Area const& area) {
    auto const& tags = area.tags();
    if (tags.has_tag("amenity", "parking")) {
      add_parking(calc_center(*area.cbegin<osmium::OuterRing>()), tags);
    }
  }

private:
  void add_parking(osmium::geom::Coordinates const& coord,
                   osmium::TagList const& tags) {
    if (!access_allowed(tags)) {
      return;
    }
    auto const fee = tags.has_key("fee") && !tags.has_tag("fee", "no");
    out_ << coord.y << "," << coord.x << "," << (fee ? "1" : "0") << "\n";
    parkings_.emplace_back(++id_, geo::latlng(coord.y, coord.x), fee);
  }

  static inline bool access_allowed(osmium::TagList const& tags) {
    auto const access = tags["access"];
    if (access == nullptr) {
      return true;
    }
    return strcmp(access, "yes") == 0 || strcmp(access, "permissive") == 0 ||
           strcmp(access, "public") == 0;
  }

  std::ofstream out_;
  std::vector<parking_lot>& parkings_;
  int32_t id_{0};
};

bool extract_parkings(std::string const& osm_file,
                      std::string const& parking_file,
                      std::vector<parking_lot>& parkings) {
  auto const out_dir = fs::path{parking_file}.parent_path();
  if (!out_dir.empty()) {
    fs::create_directories(out_dir);
  }

  scoped_timer timer("Extracting parking data");

  osmium::io::File const input_file{osm_file};

  osmium::area::Assembler::config_type assembler_config;
  assembler_config.create_empty_areas = false;
  osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{
      assembler_config};

  std::cout << "Pass 1..." << std::endl;
  osmium::relations::read_relations(input_file, mp_manager);

  index_type index;
  location_handler_type location_handler{index};
  parking_handler data_handler{parking_file, parkings};

  std::cout << "Pass 2..." << std::endl;
  osmium::io::Reader reader{input_file, osmium::io::read_meta::no};
  osmium::apply(reader, location_handler, data_handler,
                mp_manager.handler(
                    [&data_handler](const osmium::memory::Buffer& area_buffer) {
                      osmium::apply(area_buffer, data_handler);
                    }));

  reader.close();

  return true;
}

}  // namespace motis::parking::prepare
