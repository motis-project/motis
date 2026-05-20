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
  auto const mask = query.modes_.transform(to_clasz_mask)
                        .value_or(n::routing::all_clasz_allowed());
  auto const grouped = query.grouped_.value_or(false);

  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}", query.max_);

  auto const max_results = config_.get_limits().stops_max_results_;
  auto res = api::stops_response{};

  // --- Ungrouped ---
  if (!grouped) {
    loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
      auto location_clasz_mask = n::routing::clasz_mask_t{0};
      for (auto const r : tt_.location_routes_[l]) {
        location_clasz_mask |= n::routing::to_mask(tt_.route_clasz_[r]);
      }
      if (location_clasz_mask == 0) {
        return;
      }

      auto p = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                        query.language_, tt_location{l});
      p.modes_ = to_modes(location_clasz_mask, 5);

      utl::verify<net::too_many_exception>(res.size() < max_results,
                                           "too many stops");
      res.emplace_back(std::move(p));
    });
    return res;
  }

  // --- Grouped ---
  if (ae_ == nullptr) {
    throw net::bad_request_exception(
        "grouped stops requires geocoding data (adr_extend)");
  }

  // Group all matching locations by their place.
  auto seen_places =
      hash_map<adr_extra_place_idx_t, std::vector<n::location_idx_t>>{};
  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    if (tt_.location_routes_[l].empty()) {
      return;  // Skip locations not served by any route.
    }
    auto const place_idx = ae_->location_place_[l];
    auto const clasz = ae_->place_clasz_[place_idx];
    if ((mask & clasz) != 0U) {
      seen_places[place_idx].push_back(l);
    }
  });

  // Generate one place per group, using the center coordinate
  // and the union of all modes of the grouped locations.
  for (auto const& [_, locations] : seen_places) {
    auto center_lat = 0.0;
    auto center_lng = 0.0;
    for (auto const l : locations) {
      center_lat += tt_.locations_.coordinates_[l].lat_;
      center_lng += tt_.locations_.coordinates_[l].lng_;
    }

    auto p = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                      query.language_, tt_location{locations.front()});
    p.lat_ = center_lat / static_cast<double>(locations.size());
    p.lon_ = center_lng / static_cast<double>(locations.size());

    utl::verify<net::too_many_exception>(res.size() < max_results,
                                         "too many stops");
    res.emplace_back(std::move(p));
  }

  return res;
}

}  // namespace motis::ep
