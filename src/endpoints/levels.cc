#include "motis/endpoints/levels.h"

#include "utl/pipes/all.h"
#include "utl/pipes/vec.h"
#include "utl/to_vec.h"

#include "osr/lookup.h"

#include "motis/parse_location.h"
#include "motis/types.h"

namespace json = boost::json;

namespace motis::ep {

api::levels_response levels::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::levels_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  utl::verify(min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify(max.has_value(), "max not a coordinate: {}", query.max_);
  auto levels = hash_set<float>{};
  l_.find({min->pos_, max->pos_}, [&](osr::way_idx_t const x) {
    auto const p = w_.r_->way_properties_[x];
    levels.emplace(to_float(p.from_level()));
    levels.emplace(to_float(p.to_level()));
  });
  auto levels_sorted =
      utl::to_vec(levels, [](float const l) { return static_cast<double>(l); });
  utl::sort(levels_sorted, [](auto&& a, auto&& b) { return a > b; });
  return levels_sorted;
}

}  // namespace motis::ep