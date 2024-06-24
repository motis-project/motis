#include "icc/endpoints/matches.h"

#include "osr/geojson.h"

#include "icc/location_routes.h"
#include "icc/match_platforms.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

std::string get_names(osr::platforms const& pl, osr::platform_idx_t const x) {
  auto ss = std::stringstream{};
  for (auto const& y : pl.platform_names_[x]) {
    ss << y.view() << ", ";
  }
  return ss.str();
}

json::value matches::operator()(json::value const& query) const {
  auto const& q = query.as_array();

  auto const min = geo::latlng{q[1].as_double(), q[0].as_double()};
  auto const max = geo::latlng{q[3].as_double(), q[2].as_double()};

  auto matches = json::array{};

  pl_.find(min, max, [&](osr::platform_idx_t const p) {
    auto const center = get_platform_center(pl_, w_, p);
    if (!center.has_value()) {
      return;
    }
    matches.emplace_back(json::value{
        {"type", "Feature"},
        {"properties",
         {{"type", "platform"},
          {"level", to_float(pl_.get_level(w_, p))},
          {"platform_names", fmt::format("{}", get_names(pl_, p))}}},
        {"geometry", osr::to_point(osr::point::from_latlng(*center))}});
  });

  loc_rtree_.find(min, max, [&](n::location_idx_t const l) {
    auto const pos = tt_.locations_.coordinates_[l];
    auto const match = get_match(tt_, pl_, w_, l);
    auto props =
        json::value{{"name", tt_.locations_.names_[l].view()},
                    {"id", tt_.locations_.ids_[l].view()},
                    {"src", to_idx(tt_.locations_.src_[l])},
                    {"type", "location"},
                    {"trips", fmt::format("{}", get_location_routes(tt_, l))}}
            .as_object();
    if (match == osr::platform_idx_t::invalid()) {
      props.emplace("level", "-");
    } else {
      std::visit(
          utl::overloaded{
              [&](osr::way_idx_t x) {
                props.emplace("osm_way_id", to_idx(w_.way_osm_idx_[x]));
                props.emplace("level",
                              to_float(w_.r_->way_properties_[x].from_level()));
              },
              [&](osr::node_idx_t x) {
                props.emplace("osm_node_id", to_idx(w_.node_to_osm_[x]));
                props.emplace(
                    "level", to_float(w_.r_->node_properties_[x].from_level()));
              }},
          osr::to_ref(pl_.platform_ref_[match][0]));
    }
    matches.emplace_back(
        json::value{{"type", "Feature"},
                    {"properties", props},
                    {"geometry", osr::to_point(osr::point::from_latlng(pos))}});

    if (match == osr::platform_idx_t::invalid()) {
      return;
    }

    props.emplace("platform_names", fmt::format("{}", get_names(pl_, match)));

    auto const center = get_platform_center(pl_, w_, match);
    if (!center.has_value()) {
      return;
    }

    props.insert_or_assign("type", "match");
    matches.emplace_back(json::value{
        {"type", "Feature"},
        {"properties", props},
        {"geometry", osr::to_line_string({osr::point::from_latlng(*center),
                                          osr::point::from_latlng(pos)})}});
  });
  return json::value{{"type", "FeatureCollection"}, {"features", matches}};
}

}  // namespace icc::ep