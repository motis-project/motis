#include "motis/transfers/platform/platform.h"

#include "fmt/core.h"

namespace motis::transfers {

char get_osm_type_as_char(osm_type const type) {
  switch (type) {
    case osm_type::kNode: return 'n';
    case osm_type::kWay: return 'w';
    case osm_type::kRelation: return 'r';
    default: return '_';
  }
}

string to_key(platform const& pf) {
  return {fmt::format("{}:{}", std::to_string(pf.osm_id_),
                      get_osm_type_as_char(pf.osm_type_))};
}

bool operator==(platform const& a, platform const& b) {
  return a.osm_id_ == b.osm_id_ && a.osm_type_ == b.osm_type_;
};

}  // namespace motis::transfers