#pragma once

#include <string>

#include "cista/reflection/comparable.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

struct osm_way {
  CISTA_COMPARABLE();

  bool is_valid() const { return !ids_.empty(); }
  void invalidate() { ids_.clear(); }

  int64_t from() const { return path_.osm_node_ids_.front(); }
  int64_t to() const { return path_.osm_node_ids_.back(); }

  mcd::vector<int64_t> ids_;
  bool oneway_{false};
  osm_path path_;
};

mcd::vector<osm_way> aggregate_osm_ways(mcd::vector<osm_way> osm_ways);

}  // namespace motis::path
