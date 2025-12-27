#include "motis/endpoints/map/stops.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "osr/geojson.h"

#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

api::stops_response stops::operator()(boost::urls::url_view const& url) const {
  auto const query = api::stops_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}request_exception", query.max_);
  auto res = api::stops_response{};

  auto const max_results = config_.limits_.value().stops_max_results_;
  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    utl::verify<net::too_many_exception>(res.size() < max_results,
                                         "too many items");
    res.emplace_back(to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_,
                              query.language_, tt_location{l}));
  });
  return res;
}

}  // namespace motis::ep