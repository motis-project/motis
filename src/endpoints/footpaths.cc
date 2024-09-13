#include "icc/endpoints/footpaths.h"

#include "osr/geojson.h"
#include "osr/routing/route.h"

#include "utl/pipes/transform.h"
#include "utl/pipes/all.h"
#include "utl/pipes/vec.h"

#include "icc/constants.h"
#include "icc/elevators/elevators.h"
#include "icc/elevators/match_elevator.h"
#include "icc/get_loc.h"
#include "icc/match_platforms.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

json::value to_json(osr::location const& loc) {
  return json::value{{"lat", loc.pos_.lat_},
                     {"lng", loc.pos_.lng_},
                     {"level", to_float(loc.lvl_)}};
}

json::value to_json(n::timetable const& tt, n::location_idx_t const l) {
  return json::value{{"name", tt.locations_.names_[l].view()},
                     {"id", tt.locations_.ids_[l].view()},
                     {"src", to_idx(tt.locations_.src_[l])}};
}

struct fp {
  void set(osr::search_profile const p,
           n::duration_t const d,
           bool const uses_elevator) {
    switch (p) {
      case osr::search_profile::kFoot: foot_ = d; break;
      case osr::search_profile::kWheelchair:
        wheelchair_ = d;
        wheelchair_uses_elevator_ = uses_elevator;
        break;
      default: std::unreachable();
    }
  }
  std::optional<n::duration_t> default_, foot_, wheelchair_;
  bool wheelchair_uses_elevator_;
};

json::value footpaths::operator()(json::value const& query) const {
  auto const rt = rt_;
  auto const e = rt->e_.get();

  auto const q = query.as_object();
  auto const l =
      tt_.locations_
          .get({std::string_view{q.at("id").as_string()},
                n::source_idx_t{
                    q.at("src").to_number<n::source_idx_t::value_t>()}})
          .l_;

  auto neighbors = std::vector<n::location_idx_t>{};
  loc_rtree_.in_radius(
      tt_.locations_.coordinates_[l], kMaxDistance,
      [&](n::location_idx_t const l) { neighbors.emplace_back(l); });

  auto footpaths = hash_map<n::location_idx_t, fp>{};

  for (auto const fp : tt_.locations_.footpaths_out_[0][l]) {
    if (tt_.location_routes_[fp.target()].empty()) {
      continue;
    }
    footpaths[fp.target()].default_ = fp.duration();
  }

  auto const loc = get_loc(tt_, w_, pl_, matches_, l);
  for (auto const mode :
       {osr::search_profile::kFoot, osr::search_profile::kWheelchair}) {
    auto const results = osr::route(
        w_, l_, mode, loc,
        utl::to_vec(
            neighbors,
            [&](auto&& l) { return get_loc(tt_, w_, pl_, matches_, l); }),
        kMaxDuration, osr::direction::kForward, kMaxMatchingDistance,
        &e->blocked_, [](osr::path const& p) { return p.uses_elevator_; });

    for (auto const [n, r] : utl::zip(neighbors, results)) {
      if (r.has_value()) {
        auto const duration = n::duration_t{static_cast<n::duration_t::rep>(
            std::ceil(r->cost_ * kTransferTimeMultiplier / 60U))};
        if (duration < n::footpath::kMaxDuration) {
          footpaths[n].set(mode, duration, r->uses_elevator_);
        }
      }
    }
  }

  return json::value{
      {"id", to_json(tt_, l)},
      {"loc", to_json(loc)},
      {"footpaths",
       utl::all(footpaths)  //
           | utl::transform([&](auto&& fp) {
               auto const& [to, durations] = fp;
               auto const to_loc = get_loc(tt_, w_, pl_, matches_, to);
               auto val = json::value{{"id", to_json(tt_, to)},
                                      {"loc", to_json(to_loc)}}
                              .as_object();
               if (durations.default_) {
                 val.emplace("default", durations.default_->count());
               }
               if (durations.foot_) {
                 val.emplace("foot", durations.foot_->count());
               }
               if (durations.wheelchair_) {
                 val.emplace("wheelchair", durations.wheelchair_->count());
                 val.emplace("wheelchair_uses_elevator",
                             durations.wheelchair_uses_elevator_);
               }
               return val;
             })  //
           | utl::emplace_back_to<json::array>()}};
}

}  // namespace icc::ep
