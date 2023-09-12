#pragma once

#include <cstdint>
#include <vector>

#include "motis/footpaths/types.h"

#include "geo/latlng.h"

#include "cista/containers/string.h"

#include "ppr/routing/input_location.h"

namespace motis::footpaths {

enum class osm_type { kNode, kWay, kRelation, kUnknown };

// Returns the char representation of the given `osm_type`.
// n: kNode, w: kWay, r: kRelation, u: kUnknown
char get_osm_type_as_char(osm_type const);

struct platform {
  geo::latlng loc_;
  std::int64_t osm_id_{-1};
  osm_type osm_type_{osm_type::kNode};
  strings names_;
  bool is_bus_stop_{false};
};
using platforms = std::vector<platform>;

// Returns a string representation of the given platform.
string to_key(platform const&);

// platform equal operator
bool operator==(platform const& a, platform const& b);

}  // namespace motis::footpaths