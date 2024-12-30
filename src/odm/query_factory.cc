#include "motis/odm/query_factory.h"

#include "motis/endpoints/routing.h"

namespace motis::odm {

namespace n = nigiri;

static constexpr auto const kEmpty = std::vector<n::routing::offset>{};

n::routing::query query_factory::make(
    std::vector<n::routing::offset> const& start,
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>> const&
        td_start,
    std::vector<n::routing::offset> const& dest,
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>> const&
        td_dest) const {
  auto q =
      n::routing::query{.start_time_ = start_time_,
                        .start_match_mode_ = start_match_mode_,
                        .dest_match_mode_ = dest_match_mode_,
                        .use_start_footpaths_ = use_start_footpaths_,
                        .start_ = start,
                        .destination_ = dest,
                        .td_start_ = td_start,
                        .td_dest_ = td_dest,
                        .max_transfers_ = max_transfers_,
                        .max_travel_time_ = max_travel_time_,
                        .min_connection_count_ = min_connection_count_,
                        .extend_interval_earlier_ = extend_interval_earlier_,
                        .extend_interval_later_ = extend_interval_later_,
                        .prf_idx_ = prf_idx_,
                        .allowed_claszes_ = allowed_claszes_,
                        .require_bike_transport_ = require_bike_transport_,
                        .transfer_time_settings_ = transfer_time_settings_,
                        .via_stops_ = via_stops_,
                        .fastest_direct_ = fastest_direct_};
  motis::ep::remove_slower_than_fastest_direct(q);
  return q;
}

n::routing::query query_factory::walk_walk() const {
  return make(start_walk_, td_start_walk_, dest_walk_, td_dest_walk_);
}

n::routing::query query_factory::walk_short() const {
  return make(start_walk_, td_start_walk_, kEmpty, odm_dest_short_);
}

n::routing::query query_factory::walk_long() const {
  return make(start_walk_, td_start_walk_, kEmpty, odm_dest_long_);
}

n::routing::query query_factory::short_walk() const {
  return make(kEmpty, odm_start_short_, dest_walk_, td_dest_walk_);
}

n::routing::query query_factory::long_walk() const {
  return make(kEmpty, odm_start_long_, dest_walk_, td_dest_walk_);
}

n::routing::query query_factory::short_short() const {
  return make(kEmpty, odm_start_short_, kEmpty, odm_dest_short_);
}

n::routing::query query_factory::short_long() const {
  return make(kEmpty, odm_start_short_, kEmpty, odm_dest_long_);
}

n::routing::query query_factory::long_short() const {
  return make(kEmpty, odm_start_long_, kEmpty, odm_dest_short_);
}

n::routing::query query_factory::long_long() const {
  return make(kEmpty, odm_start_long_, kEmpty, odm_dest_long_);
}

}  // namespace motis::odm