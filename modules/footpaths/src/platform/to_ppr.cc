#include "motis/footpaths/platform/to_ppr.h"

namespace pr = ::ppr::routing;

namespace motis::footpaths {

pr::osm_namespace to_ppr_osm_type(osm_type const& type) {
  switch (type) {
    case osm_type::kNode: return pr::osm_namespace::NODE;
    case osm_type::kWay: return pr::osm_namespace::WAY;
    case osm_type::kRelation: return pr::osm_namespace::RELATION;
    default: return pr::osm_namespace::NODE;
  }
}

pr::input_location to_input_location(platform const& pf) {
  pr::input_location il;

  // TODO (Carsten) OSM_ELEMENT LEVEL missing
  il.osm_element_ = {pf.osm_id_, to_ppr_osm_type(pf.osm_type_)};
  il.location_ = ::ppr::make_location(pf.loc_.lng_, pf.loc_.lat_);

  return il;
}

}  // namespace motis::footpaths