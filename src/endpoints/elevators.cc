#include "icc/endpoints/elevators.h"

#include "osr/geojson.h"

#include "icc/elevators/match_elevator.h"

namespace json = boost::json;

namespace icc::ep {

json::value elevators::operator()(json::value const& query) const {
  auto const& q = query.as_array();

  auto const min = geo::latlng{q[1].as_double(), q[0].as_double()};
  auto const max = geo::latlng{q[3].as_double(), q[2].as_double()};

  auto matches = json::array{};
  elevators_rtree_.find(min, max, [&](elevator_idx_t const i) {
    auto const& e = elevators_[i];
    matches.emplace_back(json::value{
        {"type", "Feature"},
        {"properties",
         {{"type", "api"},
          {"id", e.id_},
          {"desc", e.desc_},
          {"status", (e.status_ == status::kActive ? "ACTIVE" : "INACTIVE")}}},
        {"geometry", osr::to_point(osr::point::from_latlng(e.pos_))}});
  });

  for (auto const n : l_.find_elevators(min, max)) {
    auto const match = match_elevator(elevators_rtree_, elevators_, w_, n);
    auto const pos = w_.get_node_pos(n);
    if (match != elevator_idx_t::invalid()) {
      auto const& e = elevators_[match];
      matches.emplace_back(json::value{
          {"type", "Feature"},
          {"properties",
           {{"type", "match"},
            {"osm_node_id", to_idx(w_.node_to_osm_[n])},
            {"id", e.id_},
            {"desc", e.desc_},
            {"status", e.status_ == status::kActive ? "ACTIVE" : "INACTIVE"}}},
          {"geometry",
           osr::to_line_string({pos, osr::point::from_latlng(e.pos_)})}});
    }
  }

  return json::value{{"type", "FeatureCollection"}, {"features", matches}};
}

}  // namespace icc::ep