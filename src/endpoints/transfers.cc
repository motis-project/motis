#include "motis/endpoints/transfers.h"

#include "osr/geojson.h"
#include "osr/routing/route.h"

#include "utl/pipes/all.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "motis/constants.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/match_elevator.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

api::transfers_response transfers::operator()(
    boost::urls::url_view const& url) const {
  auto const q = motis::api::transfers_params{url.params()};
  auto const rt = rt_;
  auto const e = rt->e_.get();
  auto const l = tags_.get_location(tt_, q.id_);

  auto const neighbors =
      loc_rtree_.in_radius(tt_.locations_.coordinates_[l], kMaxDistance);

  auto footpaths = hash_map<n::location_idx_t, api::Transfer>{};

  for (auto const fp : tt_.locations_.footpaths_out_[0].at(l)) {
    footpaths[fp.target()].default_ = fp.duration().count();
  }
  if (!tt_.locations_.footpaths_out_[n::kFootProfile].empty()) {
    for (auto const fp : tt_.locations_.footpaths_out_[n::kFootProfile].at(l)) {
      footpaths[fp.target()].foot_ = fp.duration().count();
    }
  }
  if (!tt_.locations_.footpaths_out_[n::kWheelchairProfile].empty()) {
    for (auto const fp :
         tt_.locations_.footpaths_out_[n::kWheelchairProfile].at(l)) {
      footpaths[fp.target()].wheelchair_ = fp.duration().count();
    }
  }
  if (!tt_.locations_.footpaths_out_[n::kCarProfile].empty()) {
    for (auto const fp : tt_.locations_.footpaths_out_[n::kCarProfile].at(l)) {
      footpaths[fp.target()].car_ = fp.duration().count();
    }
  }

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

  return {.place_ = to_place(l),
          .root_ = to_place(tt_.locations_.get_root_idx(l)),
          .equivalences_ = utl::to_vec(
              tt_.locations_.equivalences_[l],
              [&](n::location_idx_t const eq) { return to_place(eq); }),
          .hasFootTransfers_ =
              !tt_.locations_.footpaths_out_[n::kFootProfile].empty(),
          .hasWheelchairTransfers_ =
              !tt_.locations_.footpaths_out_[n::kWheelchairProfile].empty(),
          .hasCarTransfers_ =
              !tt_.locations_.footpaths_out_[n::kCarProfile].empty(),
          .transfers_ = utl::to_vec(footpaths, [&](auto&& e) {
            e.second.to_ = to_place(e.first);
            return e.second;
          })};
}

}  // namespace motis::ep
