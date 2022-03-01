#pragma once

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/ridesharing/dbschema/DBLiftKeys_generated.h"
#include "motis/ridesharing/dbschema/DBLift_generated.h"
#include "motis/ridesharing/dbschema/RoutingTable_generated.h"
#include "motis/ridesharing/lift.h"
#include "motis/ridesharing/routing_result.h"

#include <memory>
#include <optional>
#include <vector>

namespace motis::ridesharing {

using persistable_lift = typed_flatbuffer<DBLift>;
using persistable_routing_table = typed_flatbuffer<RoutingTable>;
using persistable_liftkeys = typed_flatbuffer<DBLiftKeys>;

struct database {
  explicit database(std::string const& path, size_t max_size);
  ~database();

  database(database const&) = delete;
  database& operator=(database const&) = delete;

  database(database&&) = default;
  database& operator=(database&&) = default;

  bool is_initialized() const;

  std::optional<lift> get_lift(lift_key const& key) const;
  void put_lift(persistable_lift const& lift, lift_key const& key);
  bool remove_lift(lift_key const& key);
  std::vector<lift> get_lifts() const;

  long get_station_hashcode() const;

  std::vector<std::vector<routing_result>> get_routing_table() const;
  void put_routing_table(persistable_routing_table const& routing_table,
                         long const hashcode);

  struct database_impl;
  std::unique_ptr<database_impl> impl_;
};

lift from_db(DBLift const*);

persistable_routing_table make_routing_table(
    std::vector<std::vector<routing_result>> const& routing_matrix);

persistable_lift make_db_lift(lift const&);

}  // namespace motis::ridesharing
