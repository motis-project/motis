#include "motis/osr/one_to_many_searches.h"

#include "utl/lookup.h"

namespace n = nigiri;

namespace motis {

precomputed_route one_to_many_view::find(n::location_idx_t const leg_from,
                                         n::location_idx_t const leg_to,
                                         transport_mode_t const mode,
                                         n::location_idx_t const target) const {
  if (searches_ == nullptr) {
    return {};
  }

  auto const flip = [](n::special_station const s) {
    return s == n::special_station::kStart ? n::special_station::kEnd
                                           : n::special_station::kStart;
  };

  auto const* const side = [&]() -> one_to_many_side const* {
    for (auto const x :
         {n::special_station::kStart, n::special_station::kEnd}) {
      auto const l = n::get_special_station(x);
      if (leg_from == l || leg_to == l) {
        return &(*searches_)[arrive_by_ ? flip(x) : x];
      }
    }
    return nullptr;  // not an access/egress leg
  }();

  if (side == nullptr) {
    return {};
  }

  auto const search_idx = utl::lookup(side->by_mode_, mode);
  if (!search_idx.has_value()) {
    return {};
  }

  auto const& s = side->searches_[*search_idx];
  auto const d = utl::lookup(s.dest_idx_, target);
  if (!d.has_value()) {
    return {};
  }

  return {s.state_.get(), *d, &s.flex_additional_nodes_};
}

}  // namespace motis
