#include "motis/osr/one_to_many_searches.h"

namespace n = nigiri;

namespace motis {

namespace {
// The place a set of offsets was computed from: the journey's start or its
// destination, which nigiri addresses as these two special stations.
constexpr nigiri::special_station flip(nigiri::special_station const s) {
  return s == nigiri::special_station::kStart ? nigiri::special_station::kEnd
                                              : nigiri::special_station::kStart;
}
}  // namespace

precomputed_route one_to_many_view::find(n::location_idx_t const leg_from,
                                         n::location_idx_t const leg_to,
                                         transport_mode_t const mode,
                                         n::location_idx_t const target) const {
  if (searches_ == nullptr) {
    return {};
  }

  auto const* const side = [&]() -> one_to_many_side const* {
    for (auto const x :
         {n::special_station::kStart, n::special_station::kEnd}) {
      auto const l = n::get_special_station(x);
      if (leg_from == l || leg_to == l) {
        return &(*searches_)[flipped_ ? flip(x) : x];
      }
    }
    return nullptr;  // not an access/egress leg
  }();
  if (side == nullptr) {
    return {};
  }

  auto const it = side->by_mode_.find(mode);
  if (it == end(side->by_mode_)) {
    return {};
  }
  auto const& s = side->searches_[it->second];
  auto const d = s.dest_idx_.find(target);
  if (d == end(s.dest_idx_)) {
    return {};
  }
  return {s.state_.get(), d->second, &s.flex_additional_nodes_};
}

}  // namespace motis
