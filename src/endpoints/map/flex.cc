#include "motis/endpoints/map/flex_locations.h"

#include "utl/to_vec.h"

#include "net/bad_request_exception.h"

#include "osr/geojson.h"

#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

json::value to_geometry(n::timetable const& tt, n::flex_area_idx_t const a) {
  auto const ring_to_json = [](auto&& r) {
    return utl::transform_to<json::array>(
        r, [](geo::latlng const& x) { return osr::to_array(x); });
  };

  auto const get_rings = [&](unsigned const i) {
    auto rings = json::array{};
    rings.emplace_back(ring_to_json(tt.flex_area_outers_[a][i]));
    for (auto const r : tt.flex_area_inners_[a][i]) {
      rings.emplace_back(ring_to_json(r));
    }
    return rings;
  };

  if (tt.flex_area_outers_[a].size() == 1U) {
    return {{"type", "Polygon"}, {"coordinates", get_rings(0U)}};
  } else {
    auto rings = json::array{};
    for (auto i = 0U; i != tt.flex_area_outers_[a].size(); ++i) {
      rings.emplace_back(get_rings(i));
    }
    return {{"type", "MultiPolygon"}, {"coordinates", rings}};
  }
}

boost::json::value flex_locations::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::stops_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);

  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}", query.max_);

  auto features = json::array{};
  tt_.flex_area_rtree_.search(
      min->pos_.lnglat_float(), max->pos_.lnglat_float(),
      [&](auto&&, auto&&, n::flex_area_idx_t const a) {
        features.emplace_back(json::value{
            {"type", "Feature"},
            {"id", tt_.strings_.get(tt_.flex_area_id_[a])},
            {"geometry", to_geometry(tt_, a)},
            {"properties",
             {{"stop_name",
               tt_.translate(query.language_, tt_.flex_area_name_[a])},
              {"stop_desc",
               tt_.translate(query.language_, tt_.flex_area_desc_[a])}}}});
        return true;
      });
  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    if (!tt_.location_location_groups_[l].empty()) {
      features.emplace_back(json::value{
          {"type", "Feature"},
          {"id", tags_.id(tt_, l)},
          {"geometry", osr::to_point(osr::point::from_latlng(
                           tt_.locations_.coordinates_[l]))},
          {"properties",
           {{"name", tt_.translate(query.language_, tt_.locations_.names_[l])},
            {"location_groups",
             utl::transform_to<json::array>(
                 tt_.location_location_groups_[l],
                 [&](n::location_group_idx_t const l) -> json::string {
                   return {tt_.translate(query.language_,
                                         tt_.location_group_name_[l])};
                 })}}}});
    }
    return true;
  });

  return {{"type", "FeatureCollection"}, {"features", std::move(features)}};
}

}  // namespace motis::ep