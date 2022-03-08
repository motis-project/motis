#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/parking/dbschema/FootEdges_generated.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

using persistable_foot_edges = typed_flatbuffer<FootEdges>;

struct database {
  explicit database(std::string const& path, bool read_only = false,
                    std::size_t max_size =
                        sizeof(void*) >= 8
                            ? static_cast<std::size_t>(1024) * 1024 * 1024 * 512
                            : 256 * 1024 * 1024);

  void put_footedges(persistable_foot_edges const& fe);
  std::optional<persistable_foot_edges> get_footedges(
      int32_t parking_id, std::string const& search_profile);

  void add_osm_parking_lots(std::vector<parking_lot>& parking_lots);

  std::vector<parking_lot> get_parking_lots();

private:
  lmdb::txn::dbi parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) const;
  lmdb::txn::dbi osm_parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) const;
  lmdb::txn::dbi parkendd_parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) const;
  lmdb::txn::dbi reachable_stations_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) const;
  lmdb::txn::dbi footedges_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) const;

  void init();

  lmdb::env mutable env_;
  bool open_;
  std::mutex mutex_;
  std::int32_t highest_parking_lot_id_{};
};

}  // namespace motis::parking
