#include "icc/endpoints/levels.h"

#include "utl/pipes/all.h"
#include "utl/pipes/vec.h"
#include "utl/to_vec.h"

#include "osr/lookup.h"

#include "icc/types.h"

namespace json = boost::json;

namespace icc::ep {

json::value levels::operator()(json::value const& query) const {
  auto const& q = query.at("waypoints").as_array();
  auto const min = geo::latlng{q[1].as_double(), q[0].as_double()};
  auto const max = geo::latlng{q[3].as_double(), q[2].as_double()};
  auto levels = hash_set<osr::level_t>{};
  l_.find({min, max}, [&](osr::way_idx_t const x) {
    auto const p = w_.r_->way_properties_[x];
    levels.emplace(p.from_level());
    if (p.from_level() != p.to_level()) {
      levels.emplace(p.to_level());
    }
  });
  auto levels_sorted =
      utl::to_vec(levels, [](osr::level_t const l) { return to_float(l); });
  utl::sort(levels_sorted, [](auto&& a, auto&& b) { return a > b; });
  return utl::all(levels_sorted) | utl::emplace_back_to<json::array>();
}

}  // namespace icc::ep