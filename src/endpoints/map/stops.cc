#include "motis/endpoints/map/stops.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "tiles/constants.h"

#include "motis/place.h"
#include "motis/display_filter.h"
#include "motis/parse_location.h"

using int_clasz = decltype(n::kNumClasses);

namespace motis::ep {

api::stops_response stops::operator()(boost::urls::url_view const& url) const {
  auto const query = api::stops_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  auto const zoom = query.zoom_.value_or(tiles::kMaxZoomLevel);
  auto const grouped = query.grouped_.value_or(false);

  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}request_exception", query.max_);

  auto res = api::stops_response{};
  auto const max_results = config_.get_limits().stops_max_results_;
  auto seen_places = hash_set<adr_extra_place_idx_t>{};

  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    utl::verify<net::too_many_exception>(res.size() < max_results,
                                         "too many items");
    auto const place_idx = ae_->location_place_[l];
    auto const cl_mask = ae_->place_clasz_[place_idx];

    auto should_show = false;
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      auto const cl = n::clasz{c};
      if (nigiri::routing::is_allowed(cl_mask, cl) &&
          should_display(cl, zoom, 0)) {
        should_show = true;
        break;
      }
    }
    if (!should_show) return;

    auto const [_, inserted] = seen_places.insert(place_idx);
    if (!grouped || inserted) {
      res.emplace_back(to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                                query.language_, tt_location{l}));
    }
  });
  return res;
}

}  // namespace motis::ep
