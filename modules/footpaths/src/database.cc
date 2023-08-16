#include "motis/footpaths/database.h"

#include <string_view>

#include "motis/core/common/logging.h"
#include "cista/serialization.h"
#include "cista/targets/buf.h"

#include "utl/enumerate.h"

namespace motis::footpaths {

constexpr auto const kPlatformsDB = "platforms";
constexpr auto const kMatchingsDB = "matchings";
constexpr auto const kTransReqsDB = "transreqs";
constexpr auto const kTransfersDB = "transfers";

inline std::string_view view(cista::byte_buf const& b) {
  return std::string_view{reinterpret_cast<char const*>(b.data()), b.size()};
}

database::database(std::string const& path, std::size_t const max_size) {
  env_.set_maxdbs(4);
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
  transreqs_dbi(txn, lmdb::dbi_flags::CREATE);
  transfers_dbi(txn, lmdb::dbi_flags::CREATE);

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

std::vector<std::size_t> database::put_platforms(platforms& pfs) {
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

platforms database::get_platforms() {
  auto pfs = platforms{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto platforms_db = platforms_dbi(txn);
  auto cur = lmdb::cursor{txn, platforms_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    pfs.emplace_back(
        cista::copy_from_potentially_unaligned<platform>(entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return pfs;
}

hash_map<string, platform> database::get_platforms_with_key() {
  auto pfs_with_key = hash_map<string, platform>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto platforms_db = platforms_dbi(txn);
  auto cur = lmdb::cursor{txn, platforms_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    pfs_with_key.insert(std::pair<string, platform>(
        string{entry->first},
        cista::copy_from_potentially_unaligned<platform>(entry->second)));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return pfs_with_key;
}

platforms database::get_matched_platforms() {
  auto pfs = platforms{};
  auto const matchings = get_matchings();

  for (auto const& [nloc_key, osm_key] : matchings) {
    auto pf = get_platform(osm_key);
    if (pf.has_value()) {
      pfs.emplace_back(pf.value());
    }
  }

  return pfs;
}

std::optional<platform> database::get_platform(string const& osm_key) {
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
    matching_results const& mrs) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto matchings_db = matchings_dbi(txn);
  auto platforms_db = platforms_dbi(txn);

  for (auto const& [idx, mr] : utl::enumerate(mrs)) {
    auto const nloc_key = to_key(mr.nloc_pos_);
    auto const osm_key = to_key(mr.pf_);

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

std::vector<std::pair<string, string>> database::get_matchings() {
  auto matchings = std::vector<std::pair<string, string>>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto matchings_db = matchings_dbi(txn);
  auto cur = lmdb::cursor{txn, matchings_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    matchings.emplace_back(string{entry->first}, string{entry->second});
    entry = cur.get(lmdb::cursor_op::NEXT);
  }
  cur.reset();
  return matchings;
}

hash_map<string, platform> database::get_loc_to_pf_matchings() {
  auto loc_pf_matchings = hash_map<string, platform>{};

  for (auto& [location, osm_key] : get_matchings()) {
    auto const pf = get_platform(osm_key);

    if (pf.has_value()) {
      loc_pf_matchings.insert(
          std::pair<string, platform>(location, pf.value()));
    }
  }

  return loc_pf_matchings;
}

std::vector<std::size_t> database::put_transfer_requests_keys(
    transfer_requests_keys const& treqs_k) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto transreqs_db = transreqs_dbi(txn);

  for (auto const& [idx, treq_k] : utl::enumerate(treqs_k)) {
    auto const treq_key = to_key(treq_k);

    if (auto const r = txn.get(transreqs_db, treq_key); r.has_value()) {
      continue;  // transfer request already in db
    }

    auto const serialized_treq = cista::serialize(treq_k);
    txn.put(transreqs_db, treq_key, view(serialized_treq));
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

transfer_requests_keys database::get_transfer_requests_keys() {
  auto treqs_k = transfer_requests_keys{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto transreqs_db = transreqs_dbi(txn);
  auto cur = lmdb::cursor{txn, transreqs_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    treqs_k.emplace_back(
        cista::copy_from_potentially_unaligned<transfer_request_keys>(
            entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return treqs_k;
}

std::vector<std::size_t> database::put_transfer_results(
    transfer_results const& trs) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto transfers_db = transfers_dbi(txn);

  for (auto const& [idx, tr] : utl::enumerate(trs)) {
    auto const tr_key = to_key(tr);

    if (auto const r = txn.get(transfers_db, tr_key); r.has_value()) {
      continue;  // transfer already in db
    }

    auto const serialized_tr = cista::serialize(tr);
    txn.put(transfers_db, tr_key, view(serialized_tr));
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

std::vector<std::size_t> database::update_transfer_results(
    transfer_results const& trs) {
  auto updated_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto transfers_db = transfers_dbi(txn);

  for (auto const& [idx, tr] : utl::enumerate(trs)) {
    auto const tr_key = to_key(tr);

    if (auto const r = txn.get(transfers_db, tr_key); !r.has_value()) {
      continue;  // transfer not in db
    }

    auto const serialized_tr = cista::serialize(tr);
    if (txn.del(transfers_db, tr_key)) {
      txn.put(transfers_db, tr_key, view(serialized_tr));
    }

    updated_indices.emplace_back(idx);
  }

  txn.commit();
  return updated_indices;
}

transfer_results database::get_transfer_results() {
  auto trs = transfer_results{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto transfers_db = transfers_dbi(txn);
  auto cur = lmdb::cursor(txn, transfers_db);
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    trs.emplace_back(
        cista::copy_from_potentially_unaligned<transfer_result>(entry->second));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return trs;
}

lmdb::txn::dbi database::platforms_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kPlatformsDB, flags);
}

lmdb::txn::dbi database::matchings_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kMatchingsDB, flags);
}

lmdb::txn::dbi database::transreqs_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kTransReqsDB, flags);
}

lmdb::txn::dbi database::transfers_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open(kTransfersDB, flags);
}

}  // namespace motis::footpaths
