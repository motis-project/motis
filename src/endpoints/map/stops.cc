#include "motis/endpoints/map/stops.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "adr/typeahead.h"

#include "motis/adr_extend_tt.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/server.h"
#include "motis/tag_lookup.h"
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
  auto const api_version = get_api_version(url);

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

      auto p =
          bwd_compat_lvl_adjust(to_place(&tt_, &tags_, w_, pl_, matches_, ae_,
                                         tz_, query.language_, tt_location{l}),
                                api_version);
      p.modes_ = to_modes(location_clasz_mask, 5);

      utl::verify<net::too_many_exception>(res.size() < max_results,
                                           "too many stops");
      res.emplace_back(std::move(p));
    });
    return res;
  }

  // --- Grouped ---
  if (ae_ == nullptr || t_ == nullptr) {
    throw net::bad_request_exception(
        "grouped stops requires geocoding data (adr_extend)");
  }

  // Query the place rtree for all places in the bounding box.
  auto const box = geo::box{min->pos_, max->pos_};
  ae_->place_rtree_.search(
      box.min_.lnglat_float(), box.max_.lnglat_float(),
      [&](auto const& pos, auto const& /*max*/,
          adr_extra_place_idx_t const extra_place_idx) {
        auto const clasz = ae_->place_clasz_[extra_place_idx];
        if ((mask & clasz) == 0U) {
          return true;
        }

        auto const adr_place_idx =
            adr::place_idx_t{t_->ext_start_ + cista::to_idx(extra_place_idx)};
        auto const osm_ids = t_->place_osm_ids_[adr_place_idx];
        if (osm_ids.empty()) {
          return true;
        }
        auto const representative = n::location_idx_t{
            static_cast<cista::base_t<n::location_idx_t>>(osm_ids.front())};

        auto p = bwd_compat_lvl_adjust(
            to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_, query.language_,
                     tt_location{representative}),
            api_version);
        p.lat_ = pos[1];
        p.lon_ = pos[0];

        utl::verify<net::too_many_exception>(res.size() < max_results,
                                             "too many stops");
        res.emplace_back(std::move(p));
        return true;
      });

  return res;
}

}  // namespace motis::ep
