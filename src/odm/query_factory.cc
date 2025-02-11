#include "motis/odm/query_factory.h"

#include "motis/endpoints/routing.h"

namespace motis::odm {

namespace n = nigiri;

std::vector<n::routing::query> query_factory::make_queries() const {
  auto queries = std::vector<n::routing::query>{};
  queries.push_back(
      make(start_walk_, td_start_walk_, dest_walk_, td_dest_walk_));
  queries.push_back(make(start_walk_, td_start_walk_, {}, odm_dest_short_));
  queries.push_back(make(start_walk_, td_start_walk_, {}, odm_dest_long_));
  queries.push_back(make({}, odm_start_short_, dest_walk_, td_dest_walk_));
  queries.push_back(make({}, odm_start_long_, dest_walk_, td_dest_walk_));
  queries.push_back(make({}, odm_start_short_, {}, odm_dest_short_));
  queries.push_back(make({}, odm_start_short_, {}, odm_dest_long_));
  queries.push_back(make({}, odm_start_long_, {}, odm_dest_short_));
  queries.push_back(make({}, odm_start_long_, {}, odm_dest_long_));
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