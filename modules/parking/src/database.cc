#include "motis/parking/database.h"

#include <sstream>
#include <string_view>
#include <variant>

#include "cista/serialization.h"

#include "utl/enumerate.h"

#include "fmt/core.h"

#include "motis/core/common/logging.h"

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

template <typename T>
inline std::string_view int_view(T const& val) {
  return std::string_view{reinterpret_cast<char const*>(&val), sizeof(val)};
}

inline std::string get_footedges_db_key(int32_t parking_id,
                                        std::string const& profile) {
  return fmt::format("{}:{}", parking_id, profile);
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

inline std::string_view get_parkendd_parking_lot_key(parking_lot const& lot) {
  auto const& info = std::get<parkendd_parking_lot_info>(lot.info_);
  return info.id_;
}

inline std::string serialize_reachable_stations(
    std::vector<std::pair<lookup_station, double>> const& st) {
  std::stringstream ss;
  for (auto const& s : st) {
    ss << s.first.tag_ << s.first.id_ << "|";
  }
  return ss.str();
}

database::database(std::string const& path, std::size_t const max_size) {
  env_.set_maxdbs(10);
  env_.set_mapsize(max_size);
  auto flags = lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOSYNC;
  env_.open(path.c_str(), flags);
  init();
}

void database::init() {
  // create databases
  auto txn = lmdb::txn{env_};
  auto parking_lots_db = parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  osm_parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  parkendd_parking_lots_dbi(txn, lmdb::dbi_flags::CREATE);
  reachable_stations_dbi(txn, lmdb::dbi_flags::CREATE);
  footedges_dbi(txn, lmdb::dbi_flags::CREATE);

  // find highest existing parking lot id
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

void database::put_footedges(
    const persistable_foot_edges& fe,
    std::vector<std::pair<lookup_station, double>> const& reachable_stations) {
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto reachable_stations_db = reachable_stations_dbi(txn);
  auto footedges_db = footedges_dbi(txn);
  auto const key = get_footedges_db_key(fe.get()->parking_id(),
                                        fe.get()->search_profile()->str());
  txn.put(footedges_db, key, fe.to_string());
  txn.put(reachable_stations_db, key,
          serialize_reachable_stations(reachable_stations));
  txn.commit();
}

std::optional<persistable_foot_edges> database::get_footedges(
    int32_t parking_id, std::string const& search_profile) {
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

std::vector<std::size_t> database::add_parking_lots(
    std::vector<parking_lot>& parking_lots) {
  auto added_indices = std::vector<std::size_t>{};
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto parking_lots_db = parking_lots_dbi(txn);
  auto osm_parking_lots_db = osm_parking_lots_dbi(txn);
  auto parkendd_parking_lots_db = parkendd_parking_lots_dbi(txn);

  auto const store_parking_lot = [&](parking_lot const& lot) {
    auto const serialized_lot = cista::serialize(lot);
    txn.put(parking_lots_db, lot.id_, view(serialized_lot));
  };

  for (auto const& [idx, lot] : utl::enumerate(parking_lots)) {
    if (lot.is_from_osm()) {
      auto const osm_key = get_osm_parking_lot_key(lot);
      if (auto const r = txn.get(osm_parking_lots_db, osm_key); r.has_value()) {
        lot.id_ = lmdb::as_int(r.value());
      } else {
        lot.id_ = ++highest_parking_lot_id_;
        store_parking_lot(lot);
        txn.put(osm_parking_lots_db, osm_key, int_view(lot.id_));
        added_indices.emplace_back(idx);
      }
    } else if (lot.is_from_parkendd()) {
      auto const parkendd_key = get_parkendd_parking_lot_key(lot);
      if (auto const r = txn.get(parkendd_parking_lots_db, parkendd_key);
          r.has_value()) {
        lot.id_ = lmdb::as_int(r.value());
      } else {
        lot.id_ = ++highest_parking_lot_id_;
        store_parking_lot(lot);
        txn.put(parkendd_parking_lots_db, parkendd_key, int_view(lot.id_));
        added_indices.emplace_back(idx);
      }
    }
  }

  txn.commit();
  return added_indices;
}

std::vector<parking_lot> database::get_parking_lots() {
  auto parking_lots = std::vector<parking_lot>{};
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto parking_lots_db = parking_lots_dbi(txn);
  auto cur = lmdb::cursor{txn, parking_lots_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);
  while (entry.has_value()) {
    parking_lots.emplace_back(
        cista::copy_from_potentially_unaligned<parking_lot>(entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }
  cur.reset();
  return parking_lots;
}

std::vector<foot_edge_task> database::get_foot_edge_tasks(
    station_lookup const& st, std::vector<parking_lot> const& parking_lots,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles) {
  auto tasks = std::vector<foot_edge_task>{};
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto reachable_stations_db = reachable_stations_dbi(txn);
  auto footedges_db = footedges_dbi(txn);

  for (auto const& [profile_name, pi] : ppr_profiles) {
    auto const& profile = pi.profile_;
    auto const walk_radius = static_cast<int>(
        std::ceil(profile.duration_limit_ * profile.walking_speed_));
    for (auto const& pl : parking_lots) {
      auto const key = get_footedges_db_key(pl.id_, profile_name);
      auto task = foot_edge_task{&pl, st.in_radius(pl.location_, walk_radius),
                                 &profile_name};
      if (auto const sr = txn.get(reachable_stations_db, key); sr.has_value()) {
        auto const reachable_stations =
            serialize_reachable_stations(task.stations_in_radius_);
        if (sr.value() == reachable_stations) {
          if (auto const sf = txn.get(footedges_db, key); sf.has_value()) {
            // already in db
            continue;
          }
        }
      }
      tasks.emplace_back(std::move(task));
    }
  }

  return tasks;
}

lmdb::txn::dbi database::parking_lots_dbi(lmdb::txn& txn,
                                          lmdb::dbi_flags const flags) {
  return txn.dbi_open(PARKING_LOTS_DB, flags | lmdb::dbi_flags::INTEGERKEY);
}

lmdb::txn::dbi database::osm_parking_lots_dbi(lmdb::txn& txn,
                                              lmdb::dbi_flags const flags) {
  return txn.dbi_open(OSM_PARKING_LOTS_DB, flags);
}

lmdb::txn::dbi database::parkendd_parking_lots_dbi(
    lmdb::txn& txn, lmdb::dbi_flags const flags) {
  return txn.dbi_open(PARKENDD_PARKING_LOTS_DB, flags);
}

lmdb::txn::dbi database::reachable_stations_dbi(lmdb::txn& txn,
                                                lmdb::dbi_flags const flags) {
  return txn.dbi_open(REACHABLE_STATIONS_DB, flags);
}

lmdb::txn::dbi database::footedges_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(FOOTEDGES_DB, flags);
}

}  // namespace motis::parking
