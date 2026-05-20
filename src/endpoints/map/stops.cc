#include "motis/endpoints/map/stops.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace n = nigiri;
namespace motis::ep {

api::stops_response stops::operator()(boost::urls::url_view const& url) const {
  auto const query = api::stops_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  auto const modes = query.modes_;
  auto const grouped = query.grouped_.value_or(false);

  auto any_allowed = [](auto const& mask, auto const& cs) -> bool {
    return (mask & cs) != 0;
  };

  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}request_exception", query.max_);

  auto res = api::stops_response{};
  auto const max_results = config_.get_limits().stops_max_results_;
  auto const mask = modes.has_value() ? to_clasz_mask(*modes)
                                      : n::routing::all_clasz_allowed();

  if (!grouped) {
    loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
      if (tt_.location_routes_[l].empty()) {
        return;  // Skip locations not served by any route.
      }
      utl::verify<net::too_many_exception>(res.size() < max_results,
                                           "too many items");
      auto const place_idx = ae_->location_place_[l];
      auto const clasz = ae_->place_clasz_[place_idx];
      if (any_allowed(mask, clasz)) {
        auto p = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                          query.language_, tt_location{l});
        // Ungrouped: report the modes of this specific location only.
        auto location_clasz_mask = n::routing::clasz_mask_t{0};
        for (auto const r : tt_.location_routes_[l]) {
          location_clasz_mask |= n::routing::to_mask(tt_.route_clasz_[r]);
        }
        p.modes_ = to_modes(location_clasz_mask, 5);
        res.emplace_back(std::move(p));
      }
    });
    return res;
  }

  // Phase 1: group all matching locations by their place.
  auto seen_places =
      hash_map<adr_extra_place_idx_t, std::vector<n::location_idx_t>>{};
  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    if (tt_.location_routes_[l].empty()) {
      return;  // Skip locations not served by any route.
    }
    auto const place_idx = ae_->location_place_[l];
    auto const clasz = ae_->place_clasz_[place_idx];
    if (any_allowed(mask, clasz)) {
      seen_places[place_idx].push_back(l);
    }
  });

  // Phase 2: emit one place per group, using the center coordinate and the
  // union of all modes of the grouped locations.
  for (auto const& [_, locations] : seen_places) {
    utl::verify<net::too_many_exception>(res.size() < max_results,
                                         "too many items");

    auto center_lat = 0.0;
    auto center_lng = 0.0;
    auto clasz_mask = n::routing::clasz_mask_t{0};
    for (auto const l : locations) {
      auto const pos = tt_.locations_.coordinates_[l];
      center_lat += pos.lat_;
      center_lng += pos.lng_;
      clasz_mask |= ae_->place_clasz_[ae_->location_place_[l]];
    }
    auto const count = static_cast<double>(locations.size());

    auto p = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                      query.language_, tt_location{locations.front()});
    p.lat_ = center_lat / count;
    p.lon_ = center_lng / count;
    p.modes_ = to_modes(clasz_mask, 5);
    res.emplace_back(std::move(p));
  }

  return res;
}

}  // namespace motis::ep
