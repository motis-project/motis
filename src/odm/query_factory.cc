#include "motis/odm/query_factory.h"

namespace motis::odm {

namespace n = nigiri;

n::routing::query query_factory::walk_walk() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::walk_short() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::walk_long() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::short_walk() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::long_walk() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::short_short() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::short_long() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::long_short() const {
  return n::routing::query{

  };
}

n::routing::query query_factory::long_long() const {
  return n::routing::query{

  };
}

std::vector<n::routing::query> query_factory::get_meta_routing_queries() const {
  auto queries = std::vector<n::routing::query>{};
  queries.emplace_back(walk_walk());
  queries.emplace_back(walk_short());
  queries.emplace_back(walk_long());
  queries.emplace_back(short_walk());
  queries.emplace_back(long_walk());
  queries.emplace_back(short_short());
  queries.emplace_back(short_long());
  queries.emplace_back(long_short());
  queries.emplace_back(long_long());
  return queries;
}

}  // namespace motis::odm