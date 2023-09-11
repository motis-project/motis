#include "motis/footpaths/platform/platform.h"

namespace motis::footpaths {

char get_osm_type_as_char(osm_type const type) {
  switch (type) {
    case osm_type::kNode: return 'n';
    case osm_type::kWay: return 'w';
    case osm_type::kRelation: return 'a';
    case osm_type::kUnknown: return 'u';
    default: return '_';
  }
}

bool operator==(platform const& a, platform const& b) {
  return a.osm_id_ == b.osm_id_ && a.osm_type_ == b.osm_type_;
};

}  // namespace motis::footpaths