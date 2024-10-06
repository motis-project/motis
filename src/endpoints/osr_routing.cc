#include "motis/endpoints/osr_routing.h"

#include "utl/pipes.h"

#include "osr/geojson.h"
#include "osr/routing/route.h"

#include "motis/data.h"

namespace json = boost::json;

namespace motis::ep {

osr::location parse_location(json::value const& v) {
  auto const& obj = v.as_object();
  return {obj.at("lat").as_double(), obj.at("lng").as_double(),
          obj.contains("level")
              ? osr::to_level(obj.at("level").to_number<float>())
              : osr::level_t::invalid()};
}

json::value osr_routing::operator()(json::value const& query) const {
  auto const rt = rt_;
  auto const e = rt->e_.get();

  auto const& q = query.as_object();
  auto const profile_it = q.find("profile");
  auto const profile =
      osr::to_profile(profile_it == q.end() || !profile_it->value().is_string()
                          ? to_str(osr::search_profile::kFoot)
                          : profile_it->value().as_string());
  auto const direction_it = q.find("direction");
  auto const dir = osr::to_direction(direction_it == q.end() ||
                                             !direction_it->value().is_string()
                                         ? to_str(osr::direction::kForward)
                                         : direction_it->value().as_string());
  auto const from = parse_location(q.at("start"));
  auto const to = parse_location(q.at("destination"));
  auto const max_it = q.find("max");
  auto const max = static_cast<osr::cost_t>(
      max_it == q.end() ? 3600 : max_it->value().as_int64());
  auto const p = route(w_, l_, profile, from, to, max, dir, 8,
                       e == nullptr ? nullptr : &e->blocked_);
  return p.has_value()
             ? json::value{{"type", "FeatureCollection"},
                           {"metadata",
                            {{"duration", p->cost_},
                             {"distance", p->dist_},
                             {"uses_elevator", p->uses_elevator_}}},
                           {"features",
                            utl::all(p->segments_)  //
                                |
                                utl::transform([&](auto&& s) {
                                  return json::value{
                                      {"type", "Feature"},
                                      {"properties",
                                       {{"level", to_float(s.from_level_)},
                                        {"way",
                                         s.way_ == osr::way_idx_t::invalid()
                                             ? 0U
                                             : to_idx(
                                                   w_.way_osm_idx_[s.way_])}}},
                                      {"geometry",
                                       osr::to_line_string(s.polyline_)}};
                                })  //
                                | utl::emplace_back_to<json::array>()}}
             : json::value{{"error", "no path found"}};
}

}  // namespace motis::ep