#include "motis/parking/database.h"

#include <string_view>
#include <variant>

#include "boost/filesystem.hpp"

#include "cista/serialization.h"

#include "fmt/core.h"

#include "motis/core/common/logging.h"

namespace fs = boost::filesystem;
using namespace motis::logging;

namespace motis::parking {

constexpr auto const PARKING_LOTS_DB = "parking_lots";
constexpr auto const OSM_PARKING_LOTS_DB = "osm_parking_lots";
constexpr auto const PARKENDD_PARKING_LOTS_DB = "parkendd_parking_lots";
constexpr auto const REACHABLE_STATIONS_DB = "reachable_stations";
constexpr auto const FOOTEDGES_DB = "footedges";

inline std::string_view view(cista::byte_buf const& b) {
  return std::string_view{reinterpret_cast<char const*>(b.data()), b.size()};
}

database::database(std::string const& path, bool const read_only,
                   std::size_t const max_size) {
  env_.set_maxdbs(8);
  env_.set_mapsize(max_size);
  auto flags = lmdb::env_open_flags::NOSUBDIR;
  if (read_only) {
    flags = flags | lmdb::env_open_flags::NOLOCK | lmdb::env_open_flags::NOTLS;
  }
  if ((path != "-" && fs::exists(path)) || !read_only) {
    env_.open(path.c_str(), flags);
    open_ = true;
    init();
  } else {
    open_ = false;
    LOG(warn) << "Database not found: " << path;
  }
}

void database::init() {
  // create databases
  auto txn = lmdb::txn{env_};
  auto parking_lots_db = parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  osm_parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  parkendd_parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  reachable_stations_dbi(txn, lmdb::dbi_flags::CREATE);
  footedges_dbi(txn, lmdb::dbi_flags::CREATE);

  // find highest existing parking lot
  auto parking_cur = lmdb::cursor{txn, parking_lots_db};
  auto const entry = parking_cur.get(lmdb::cursor_op::LAST);
  if (entry.has_value()) {
    highest_parking_lot_id_ = lmdb::as_int(entry->first);
  } else {
    highest_parking_lot_id_ = 0;
  }
  parking_cur.reset();

  txn.commit();
}

inline std::string get_footedges_db_key(int32_t parking_id,
                                        std::string const& profile) {
  return fmt::format("{}:{}", parking_id, profile);
}

void database::put_footedges(const persistable_foot_edges& fe) {
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto db = footedges_dbi(txn);
  auto const key = get_footedges_db_key(fe.get()->parking_id(),
                                        fe.get()->search_profile()->str());
  txn.put(db, key, fe.to_string());
  txn.commit();
}

std::optional<persistable_foot_edges> database::get_footedges(
    int32_t parking_id, std::string const& search_profile) {
  if (!open_) {
    return {};
  }
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto db = footedges_dbi(txn);
  auto const key = get_footedges_db_key(parking_id, search_profile);
  if (auto const r = txn.get(db, key); !r.has_value()) {
    return {};
  } else {
    return {persistable_foot_edges(*r)};
  }
}

inline char get_osm_str_type(osm_type const ot) {
  switch (ot) {
    case osm_type::NODE: return 'n';
    case osm_type::WAY: return 'w';
    case osm_type::RELATION: return 'r';
    default: return '?';
  }
}

inline std::string get_osm_parking_lot_key(parking_lot const& lot) {
  auto const& info = std::get<osm_parking_lot_info>(lot.info_);
  return fmt::format("{}:{}", get_osm_str_type(info.osm_type_), info.osm_id_);
}

void database::add_osm_parking_lots(std::vector<parking_lot>& parking_lots) {
  LOG(info) << "add_osm_parking_lots: " << parking_lots.size()
            << " parking lots";
  if (!open_) {
    // TODO(pablo): support case without db or require db
    return;
  }
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto parking_lots_db = parking_lots_dbi(txn);
  auto osm_parking_lots_db = osm_parking_lots_dbi(txn);

  for (auto& lot : parking_lots) {
    auto const osm_key = get_osm_parking_lot_key(lot);
    if (auto const r = txn.get(osm_parking_lots_db, osm_key); r.has_value()) {
      // lot.id_ = *cista::deserialize<std::int32_t>(r.value());
      lot.id_ = lmdb::as_int(r.value());
    } else {
      lot.id_ = ++highest_parking_lot_id_;
      auto const serialized_id = cista::serialize(lot.id_);  // TODO(pablo): fix
      auto const serialized_lot = cista::serialize(lot);
      txn.put(parking_lots_db, lot.id_, view(serialized_lot));
      txn.put(osm_parking_lots_db, osm_key, view(serialized_id));
    }
  }
  txn.commit();
}

std::vector<parking_lot> database::get_parking_lots() {
  auto lock = std::lock_guard{mutex_};
  auto parking_lots = std::vector<parking_lot>{};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto parking_lots_db = parking_lots_dbi(txn);
  auto cur = lmdb::cursor{txn, parking_lots_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);
  while (entry.has_value()) {
    // TODO(pablo): too many copies
    auto copy = std::string{entry->second};
    parking_lots.emplace_back(*cista::deserialize<parking_lot>(copy));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }
  cur.reset();
  return parking_lots;
}

lmdb::txn::dbi database::parking_lots_dbi(lmdb::txn& txn,
                                          lmdb::dbi_flags const flags) const {
  return txn.dbi_open(PARKING_LOTS_DB, flags | lmdb::dbi_flags::INTEGERKEY);
}

lmdb::txn::dbi database::osm_parking_lots_dbi(
    lmdb::txn& txn, lmdb::dbi_flags const flags) const {
  return txn.dbi_open(OSM_PARKING_LOTS_DB, flags);
}

lmdb::txn::dbi database::parkendd_parking_lots_dbi(
    lmdb::txn& txn, lmdb::dbi_flags const flags) const {
  return txn.dbi_open(PARKENDD_PARKING_LOTS_DB, flags);
}

lmdb::txn::dbi database::reachable_stations_dbi(
    lmdb::txn& txn, lmdb::dbi_flags const flags) const {
  return txn.dbi_open(REACHABLE_STATIONS_DB, flags);
}

lmdb::txn::dbi database::footedges_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) const {
  return txn.dbi_open(FOOTEDGES_DB, flags);
}

}  // namespace motis::parking
