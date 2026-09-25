#include "motis/endpoints/transfers.h"

#include "osr/geojson.h"
#include "osr/routing/route.h"

#include "utl/pipes/all.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "nigiri/routing/for_each_hub_source.h"

#include "motis/constants.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/match_elevator.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"
#include "motis/place.h"
#include "motis/server.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

api::transfers_response transfers::operator()(
    boost::urls::url_view const& url) const {
  auto const q = motis::api::transfers_params{url.params()};
  auto const rt = std::atomic_load(&rt_);
  auto const e = rt->e_.get();
  auto const l = tags_.get_location(tt_, q.id_);
  auto const api_version = get_api_version(url);

  auto const neighbors =
      loc_rtree_.in_radius(tt_.locations_.coordinates_[l], kMaxDistance);

  auto footpaths = hash_map<n::location_idx_t, api::Transfer>{};
  auto const add = [&](n::profile_idx_t const prf, auto&& field) {
    n::routing::for_each_transfer<n::direction::kForward>(
        tt_, nullptr, prf, l, [&](n::footpath const fp) {
          // a virtual location (transfers.txt rules) has no id of its own, it
          // is its stop outside of the routing
          if (tt_.locations_.types_[fp.target()] == n::location_type::kVirt) {
            return;
          }
          // a pair can come as a footpath and from a hub: the shortest
          auto& d = field(footpaths[fp.target()]);
          d = std::min(d.value_or(fp.duration().count()),
                       static_cast<double>(fp.duration().count()));
        });
  };
  add(0U, [](api::Transfer& t) -> auto& { return t.default_; });
  add(n::kFootProfile, [](api::Transfer& t) -> auto& { return t.foot_; });
  add(n::kWheelchairProfile,
      [](api::Transfer& t) -> auto& { return t.wheelchair_; });
  add(n::kCarProfile, [](api::Transfer& t) -> auto& { return t.car_; });

  auto const loc = get_loc(tt_, w_, pl_, matches_, l);
  for (auto const mode :
       {osr::search_profile::kFoot, osr::search_profile::kWheelchair}) {
    auto const results = osr::route(
        to_profile_parameters(mode, {}), w_, l_, mode, loc,
        utl::to_vec(
            neighbors,
            [&](auto&& l) { return get_loc(tt_, w_, pl_, matches_, l); }),
        c_.timetable_.value().max_footpath_length_ * 60U,
        osr::direction::kForward, c_.timetable_.value().max_matching_distance_,
        e == nullptr ? nullptr : &e->blocked_, nullptr, nullptr,
        [](osr::path const& p) { return p.uses_elevator_; });

    for (auto const [n, r] : utl::zip(neighbors, results)) {
      if (r.has_value()) {
        auto& fp = footpaths[n];
        auto const duration = std::ceil(r->cost_ / 60U);
        if (duration < n::footpath::kMaxDuration.count()) {
          switch (mode) {
            case osr::search_profile::kFoot: fp.footRouted_ = duration; break;
            case osr::search_profile::kWheelchair:
              fp.wheelchairRouted_ = duration;
              fp.wheelchairUsesElevator_ = r->uses_elevator_;
              break;
            default: std::unreachable();
          }
        }
      }
    }
  }

  auto const to_place = [&](n::location_idx_t const l) -> api::Place {
    return {
        .name_ =
            std::string{tt_.get_default_translation(tt_.locations_.names_[l])},
        .stopId_ = std::string{tt_.locations_.ids_[l].view()},
        .lat_ = tt_.locations_.coordinates_[l].lat(),
        .lon_ = tt_.locations_.coordinates_[l].lng(),
        .level_ = pl_.get_level(w_, matches_[l]).to_float(),
        .vertexType_ = api::VertexTypeEnum::NORMAL};
  };

  return {.place_ = bwd_compat_lvl_adjust(to_place(l), api_version),
          .root_ = bwd_compat_lvl_adjust(
              to_place(tt_.locations_.get_root_idx(l)), api_version),
          .equivalences_ = utl::to_vec(tt_.locations_.equivalences_[l],
                                       [&](n::location_idx_t const eq) {
                                         return bwd_compat_lvl_adjust(
                                             to_place(eq), api_version);
                                       }),
          .hasFootTransfers_ =
              !tt_.locations_.footpaths_out_[n::kFootProfile].empty(),
          .hasWheelchairTransfers_ =
              !tt_.locations_.footpaths_out_[n::kWheelchairProfile].empty(),
          .hasCarTransfers_ =
              !tt_.locations_.footpaths_out_[n::kCarProfile].empty(),
          .transfers_ = utl::to_vec(footpaths, [&](auto&& e) {
            e.second.to_ =
                bwd_compat_lvl_adjust(to_place(e.first), api_version);
            return e.second;
          })};
}

}  // namespace motis::ep
