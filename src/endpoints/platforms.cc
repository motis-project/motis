#include "motis/endpoints/platforms.h"

#include "osr/geojson.h"

namespace json = boost::json;

namespace motis::ep {

json::value platforms::operator()(json::value const& query) const {
  auto const& q = query.as_object();
  auto const level = q.contains("level")
                         ? osr::to_level(query.at("level").to_number<float>())
                         : osr::kNoLevel;
  auto const waypoints = q.at("waypoints").as_array();
  auto const min = osr::point::from_latlng(
      {waypoints[1].as_double(), waypoints[0].as_double()});
  auto const max = osr::point::from_latlng(
      {waypoints[3].as_double(), waypoints[2].as_double()});

  auto gj = osr::geojson_writer{.w_ = w_, .platforms_ = &pl_};
  pl_.find(min, max, [&](osr::platform_idx_t const i) {
    if (level == osr::kNoLevel || pl_.get_level(w_, i) == level) {
      gj.write_platform(i);
    }
  });

  return gj.json();
}

}  // namespace motis::ep