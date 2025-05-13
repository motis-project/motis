#include "motis/endpoints/map/stops.h"

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
  utl::verify(min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify(max.has_value(), "max not a coordinate: {}", query.max_);
  auto res = api::stops_response{};

  auto const max_results = config_.limits_.value().stops_max_results_;
  loc_rtree_.find({min->pos_, max->pos_}, [&](n::location_idx_t const l) {
    utl::verify(res.size() < max_results, "too many items");
    res.emplace_back(to_place(&tt_, &tags_, w_, pl_, matches_, tt_location{l}));
  });
  return res;
}

}  // namespace motis::ep