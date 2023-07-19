#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/ppr/profile_info.h"

#include "motis/parking/dbschema/FootEdges_generated.h"

#include "motis/parking/foot_edge_task.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

using persistable_foot_edges = typed_flatbuffer<FootEdges>;

struct database {
  explicit database(std::string const& path,
                    std::size_t max_size =
                        sizeof(void*) >= 8
                            ? static_cast<std::size_t>(1024) * 1024 * 1024 * 512
                            : 256 * 1024 * 1024);

  void put_footedges(
      persistable_foot_edges const& fe,
      std::vector<std::pair<lookup_station, double>> const& reachable_stations);

  std::optional<persistable_foot_edges> get_footedges(
      int32_t parking_id, std::string const& search_profile);

  std::vector<std::size_t> add_parking_lots(
      std::vector<parking_lot>& parking_lots);

  std::vector<parking_lot> get_parking_lots();

  std::vector<foot_edge_task> get_foot_edge_tasks(
      station_lookup const& st, std::vector<parking_lot> const& parking_lots,
      std::map<std::string, motis::ppr::profile_info> const& ppr_profiles);

private:
  static lmdb::txn::dbi parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi osm_parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi parkendd_parking_lots_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi reachable_stations_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi footedges_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  void init();

  lmdb::env mutable env_;
  std::mutex mutex_;
  std::int32_t highest_parking_lot_id_{};
};

}  // namespace motis::parking
