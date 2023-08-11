#include "motis/footpaths/database.h"

#include "fmt/core.h"

#include "utl/enumerate.h"

namespace motis::footpaths {

constexpr auto const kPlatformsDB = "platforms";
constexpr auto const kMatchingsDB = "matchings";

inline std::string_view view(cista::byte_buf const& b) {
  return std::string_view{reinterpret_cast<char const*>(b.data()), b.size()};
}

database::database(std::string const& path, std::size_t const max_size) {
  env_.set_maxdbs(2);
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

  for (auto [idx, pf] : utl::enumerate(pfs)) {
    auto const osm_key = to_key(pf);
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

std::vector<platform> database::get_matched_platforms() {
  auto pfs = std::vector<platform>{};
  auto const matchings = get_matchings();

  for (auto const& [nloc_key, osm_key] : matchings) {
    auto pf = get_platform(osm_key);
    if (pf.has_value()) {
      pfs.emplace_back(pf.value());
    }
  }

  return pfs;
}

std::optional<platform> database::get_platform(std::string const& osm_key) {
  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto platforms_db = platforms_dbi(txn);

  auto entry = txn.get(platforms_db, osm_key);

  if (entry.has_value()) {
    return cista::copy_from_potentially_unaligned<platform>(entry.value());
  }

  return {};
}

std::vector<size_t> database::put_matching_results(
    std::vector<matching_result> const& mrs) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto matchings_db = matchings_dbi(txn);
  auto platforms_db = platforms_dbi(txn);

  for (auto const& [idx, mr] : utl::enumerate(mrs)) {
    auto const nloc_key = to_key(mr.nloc_pos_);
    auto const osm_key = to_key(*mr.pf_);

    if (auto const r = txn.get(matchings_db, nloc_key); r.has_value()) {
      continue;  // nloc already matched in db
    }
    if (auto const r = txn.get(platforms_db, osm_key); !r.has_value()) {
      continue;  // osm platform not in platform_db
    }

    txn.put(matchings_db, nloc_key, osm_key);
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

std::vector<std::pair<std::string, std::string>> database::get_matchings() {
  auto matchings = std::vector<std::pair<std::string, std::string>>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto matchings_db = matchings_dbi(txn);
  auto cur = lmdb::cursor{txn, matchings_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    matchings.emplace_back(
        cista::copy_from_potentially_unaligned<std::string>(entry->first),
        cista::copy_from_potentially_unaligned<std::string>(entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return matchings;
}

hash_map<std::string, platform> database::get_loc_to_pf_matchings() {
  auto loc_pf_matchings = hash_map<std::string, platform>{};

  for (auto& [location, osm_key] : get_matchings()) {
    auto const pf = get_platform(osm_key);

    if (pf.has_value()) {
      loc_pf_matchings.insert(
          std::pair<std::string, platform>(location, pf.value()));
    }
  }

  return loc_pf_matchings;
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