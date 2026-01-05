#include "motis/endpoints/matches.h"

#include "net/too_many_exception.h"

#include "osr/geojson.h"

#include "motis/location_routes.h"
#include "motis/match_platforms.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

constexpr auto const kLimit = 2048;

std::string get_names(osr::platforms const& pl, osr::platform_idx_t const x) {
  auto ss = std::stringstream{};
  for (auto const y : pl.platform_names_[x]) {
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
    utl::verify<net::too_many_exception>(matches.size() < kLimit,
                                         "too many items");

    auto const center = get_platform_center(pl_, w_, p);
    if (!center.has_value()) {
      return;
    }
    matches.emplace_back(json::value{
        {"type", "Feature"},
        {"properties",
         {{"type", "platform"},
          {"level", pl_.get_level(w_, p).to_float()},
          {"platform_names", fmt::format("{}", get_names(pl_, p))}}},
        {"geometry", osr::to_point(osr::point::from_latlng(*center))}});
  });

  loc_rtree_.find({min, max}, [&](n::location_idx_t const l) {
    utl::verify<net::too_many_exception>(matches.size() < kLimit,
                                         "too many items");

    auto const pos = tt_.locations_.coordinates_[l];
    auto const match = get_match(tt_, pl_, w_, l);
    auto props =
        json::value{
            {"name", tt_.get_default_translation(tt_.locations_.names_[l])},
            {"id", tags_.id(tt_, l)},
            {"src", to_idx(tt_.locations_.src_[l])},
            {"type", "location"},
            {"trips", fmt::format("{}", get_location_routes(tt_, l))}}
            .as_object();
    if (match == osr::platform_idx_t::invalid()) {
      props.emplace("level", "-");
    } else {
      std::visit(utl::overloaded{
                     [&](osr::way_idx_t x) {
                       props.emplace("osm_way_id", to_idx(w_.way_osm_idx_[x]));
                       props.emplace(
                           "level",
                           w_.r_->way_properties_[x].from_level().to_float());
                     },
                     [&](osr::node_idx_t x) {
                       props.emplace("osm_node_id", to_idx(w_.node_to_osm_[x]));
                       props.emplace(
                           "level",
                           w_.r_->node_properties_[x].from_level().to_float());
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

}  // namespace motis::ep