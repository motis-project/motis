#include "motis/odm/query_factory.h"

#include "motis/endpoints/routing.h"

namespace motis::odm {

namespace n = nigiri;

std::vector<n::routing::query> query_factory::make_queries(
    bool const with_taxi, bool const with_ride_sharing) const {
  auto queries = std::vector<n::routing::query>{};
  queries.push_back(
      make(start_walk_, td_start_walk_, dest_walk_, td_dest_walk_));
  if (with_taxi) {
    if (!dest_taxi_short_.empty()) {
      queries.push_back(
          make(start_walk_, td_start_walk_, dest_walk_, dest_taxi_short_));
    }
    if (!dest_taxi_long_.empty()) {
      queries.push_back(
          make(start_walk_, td_start_walk_, dest_walk_, dest_taxi_long_));
    }
    if (!start_taxi_short_.empty()) {
      queries.push_back(
          make(start_walk_, start_taxi_short_, dest_walk_, td_dest_walk_));
    }
    if (!start_taxi_long_.empty()) {
      queries.push_back(
          make(start_walk_, start_taxi_long_, dest_walk_, td_dest_walk_));
    }
  }
  if (with_ride_sharing) {
    if (!start_ride_sharing_.empty()) {
      queries.push_back(
          make(start_walk_, start_ride_sharing_, dest_walk_, td_dest_walk_));
    }
    if (!dest_ride_sharing_.empty()) {
      queries.push_back(
          make(start_walk_, td_start_walk_, dest_walk_, dest_ride_sharing_));
    }
  }
  return queries;
}

n::routing::query query_factory::make(
    std::vector<n::routing::offset> const& start,
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>> const&
        td_start,
    std::vector<n::routing::offset> const& dest,
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>> const&
        td_dest) const {
  auto q = base_query_;
  q.start_ = start;
  q.destination_ = dest;
  q.td_start_ = td_start;
  q.td_dest_ = td_dest;
  motis::ep::remove_slower_than_fastest_direct(q);
  return q;
}

}  // namespace motis::odm