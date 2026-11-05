#include "motis/endpoints/transfers.h"

#include "osr/geojson.h"
#include "osr/routing/route.h"

#include "utl/pipes/all.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "motis/constants.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/match_elevator.h"
#include "motis/feed_labels.h"
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
  auto const blocked = labels_.blocked(tt_, nullptr, {}, {});
  auto const l = tags_.get_location(tt_, q.id_, blocked);
  auto const api_version = get_api_version(url);

  auto const is_visible = [&](n::location_idx_t const x) {
    return !blocked.srcs_.test(tt_.locations_.src_[x]);
  };
  auto neighbors =
      loc_rtree_.in_radius(tt_.locations_.coordinates_[l], kMaxDistance);
  std::erase_if(neighbors,
                [&](n::location_idx_t const x) { return !is_visible(x); });

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
        std::chrono::seconds{c_.timetable_.value().max_footpath_length_ * 60U},
        osr::direction::kForward, c_.timetable_.value().max_matching_distance_,
        e == nullptr ? nullptr : &e->blocked_, nullptr, nullptr,
        [](osr::path const& p) { return p.uses_elevator_; });

    for (auto const [n, r] : utl::zip(neighbors, results)) {
      if (r.has_value()) {
        auto& fp = footpaths[n];
        auto const duration = std::ceil(r->duration_.count() / 60.0);
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

  auto equivalences = std::vector<api::Place>{};
  for (auto const eq : tt_.locations_.equivalences_[l]) {
    if (is_visible(eq)) {
      equivalences.emplace_back(
          bwd_compat_lvl_adjust(to_place(eq), api_version));
    }
  }

  auto transfers = std::vector<api::Transfer>{};
  for (auto& [target, transfer] : footpaths) {
    if (is_visible(target)) {
      transfer.to_ = bwd_compat_lvl_adjust(to_place(target), api_version);
      transfers.emplace_back(std::move(transfer));
    }
  }

  return {.place_ = bwd_compat_lvl_adjust(to_place(l), api_version),
          .root_ = bwd_compat_lvl_adjust(
              to_place(tt_.locations_.get_root_idx(l)), api_version),
          .equivalences_ = std::move(equivalences),
          .hasFootTransfers_ =
              !tt_.locations_.footpaths_out_[n::kFootProfile].empty(),
          .hasWheelchairTransfers_ =
              !tt_.locations_.footpaths_out_[n::kWheelchairProfile].empty(),
          .hasCarTransfers_ =
              !tt_.locations_.footpaths_out_[n::kCarProfile].empty(),
          .transfers_ = std::move(transfers)};
}

}  // namespace motis::ep
