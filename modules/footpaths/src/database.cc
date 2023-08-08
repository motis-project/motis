#include "motis/footpaths/database.h"

#include "fmt/core.h"

#include "utl/enumerate.h"

namespace motis::footpaths {

constexpr auto const kPlatformsDB = "platforms";
constexpr auto const kMatchingsDB = "matchings";

inline std::string_view view(cista::byte_buf const& b) {
  return std::string_view{reinterpret_cast<char const*>(b.data()), b.size()};
}

inline std::string get_platform_key(platform const& pf) {
  return fmt::format("{}:{}", get_osm_str_type(pf.info_.osm_type_),
                     pf.info_.osm_id_);
}

inline std::string to_string(geo::latlng const& pos) {
  return fmt::format("{}:{}", pos.lat_, pos.lng_);
}

database::database(std::string const& path, std::size_t const max_size) {
  env_.set_maxdbs(1);
  env_.set_mapsize(max_size);
  auto flags = lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOSYNC;
  env_.open(path.c_str(), flags);
  init();
}

void database::init() {
  // create database
  auto txn = lmdb::txn{env_};
  auto platforms_db = platforms_dbi(txn, lmdb::dbi_flags::CREATE);
  matchings_dbi(txn, lmdb::dbi_flags::CREATE);

  // find highest platform id in db
  auto cur = lmdb::cursor{txn, platforms_db};
  auto const entry = cur.get(lmdb::cursor_op::LAST);
  highest_platform_id_ = 0;
  if (entry.has_value()) {
    highest_platform_id_ = lmdb::as_int(entry->first);
  }

  cur.reset();
  txn.commit();
}

std::vector<std::size_t> database::put_platforms(std::vector<platform>& pfs) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto platforms_db = platforms_dbi(txn);

  for (auto const& [idx, pf] : utl::enumerate(pfs)) {
    auto const osm_key = get_platform_key(pf);
    if (auto const r = txn.get(platforms_db, osm_key); r.has_value()) {
      continue;  // platform already in db
    }

    pf.id_ = ++highest_platform_id_;
    auto const serialized_pf = cista::serialize(pf);
    txn.put(platforms_db, osm_key, view(serialized_pf));
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

std::vector<platform> database::get_platforms() {
  auto platforms = std::vector<platform>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto platforms_db = platforms_dbi(txn);
  auto cur = lmdb::cursor{txn, platforms_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    platforms.emplace_back(
        cista::copy_from_potentially_unaligned<platform>(entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return platforms;
}

std::vector<size_t> database::put_matching_results(
    std::vector<matching_result>& mrs) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto matchings_db = matchings_dbi(txn);
  auto platform_db = platforms_dbi(txn);

  for (auto const& [idx, mr] : utl::enumerate(mrs)) {
    auto const nloc_key = to_string(mr.nloc_pos_);
    auto const osm_key = get_platform_key(*mr.pf_);

    if (auto const r = txn.get(matchings_db, nloc_key); r.has_value()) {
      continue;  // nloc already matched in db
    }
    if (auto const r = txn.get(platform_db, osm_key); !r.has_value()) {
      continue;  // osm platform not in platform_db
    }

    txn.put(matchings_db, nloc_key, osm_key);
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

lmdb::txn::dbi database::platforms_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kPlatformsDB, flags);
}

lmdb::txn::dbi database::matchings_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kMatchingsDB, flags);
}

}  // namespace motis::footpaths