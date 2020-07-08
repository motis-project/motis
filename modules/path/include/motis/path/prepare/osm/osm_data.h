#pragma once

#include <memory>
#include <string>

#include "cista/memory_holder.h"

#include "motis/hash_map.h"
#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/path/prepare/osm/osm_way.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

struct osm_stop_position {
  bool has_category(source_spec::category) const;

  mcd::string name_;
  mcd::vector<source_spec::category> categories_;
  int64_t id_;
  geo::latlng pos_;
};

struct osm_data {
  mcd::vector<osm_stop_position> stop_positions_;
  mcd::vector<geo::latlng> plattforms_;
  mcd::hash_map<source_spec, mcd::vector<mcd::vector<osm_way>>> profiles_;
};

mcd::unique_ptr<osm_data> parse_osm(std::string const& osm_file);

mcd::unique_ptr<osm_data> read_osm_data(std::string const& fname,
                                        cista::memory_holder&);
void write_osm_data(std::string const& fname, mcd::unique_ptr<osm_data> const&);

}  // namespace motis::path
