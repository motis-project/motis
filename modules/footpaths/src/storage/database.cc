#include "motis/footpaths/storage/database.h"

#include <string_view>

#include "cista/hashing.h"
#include "cista/serialization.h"

#include "motis/footpaths/keys.h"

#include "utl/enumerate.h"

namespace fs = std::filesystem;

namespace motis::footpaths {

constexpr auto const kProfilesDB = "profiles";
constexpr auto const kPlatformsDB = "platforms";
constexpr auto const kMatchingsDB = "matchings";
constexpr auto const kTransReqsDB = "transreqs";
constexpr auto const kTransfersDB = "transfers";

inline std::string_view view(cista::byte_buf const& b) {
  return std::string_view{reinterpret_cast<char const*>(b.data()), b.size()};
}

database::database(fs::path const& db_file_path,
                   std::size_t const db_max_size) {
  env_.set_maxdbs(5);
  env_.set_mapsize(db_max_size);
  auto flags = lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOSYNC;
  env_.open(db_file_path.c_str(), flags);
  init();
}

void database::init() {
  // create database
  auto txn = lmdb::txn{env_};
  auto profiles_db = profiles_dbi(txn, lmdb::dbi_flags::CREATE);
  platforms_dbi(txn, lmdb::dbi_flags::CREATE);
  matchings_dbi(txn, lmdb::dbi_flags::CREATE);
  transreqs_dbi(txn, lmdb::dbi_flags::CREATE);
  transfers_dbi(txn, lmdb::dbi_flags::CREATE);

  // find highes profiles id in db
  auto cur = lmdb::cursor{txn, profiles_db};
  auto entry = cur.get(lmdb::cursor_op::LAST);
  highest_profile_id_ = profile_key_t{0};
  if (entry.has_value()) {
    highest_profile_id_ =
        cista::copy_from_potentially_unaligned<profile_key_t>(entry->second);
  }

  cur.reset();
  txn.commit();
}

std::vector<std::size_t> database::put_profiles(
    std::vector<string> const& prf_names) {
  auto added_indices = std::vector<std::size_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto profiles_db = profiles_dbi(txn);

  for (auto [idx, name] : utl::enumerate(prf_names)) {
    if (auto const r = txn.get(profiles_db, name); r.has_value()) {
      continue;  // profile already in db
    }
    ++highest_profile_id_;
    auto const serialized_key = cista::serialize(highest_profile_id_);
    txn.put(profiles_db, name, view(serialized_key));
    added_indices.emplace_back(idx);
  }

  txn.commit();
  return added_indices;
}

hash_map<string, profile_key_t> database::get_profile_keys() {
  auto keys_with_name = hash_map<string, profile_key_t>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto profiles_db = profiles_dbi(txn);
  auto cur = lmdb::cursor{txn, profiles_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    keys_with_name.insert(std::pair<string, profile_key_t>(
        string{entry->first},
        cista::copy_from_potentially_unaligned<profile_key_t>(entry->second)));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return keys_with_name;
}

hash_map<profile_key_t, string> database::get_profile_key_to_name() {
  auto keys_with_name = hash_map<profile_key_t, string>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto profiles_db = profiles_dbi(txn);
  auto cur = lmdb::cursor{txn, profiles_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    keys_with_name.insert(std::pair<profile_key_t, string>(
        cista::copy_from_potentially_unaligned<profile_key_t>(entry->second),
        string{entry->first}));
    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return keys_with_name;
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

std::vector<std::pair<nlocation_key_t, string>> database::get_matchings() {
  auto matchings = std::vector<std::pair<nlocation_key_t, string>>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto matchings_db = matchings_dbi(txn);
  auto cur = lmdb::cursor{txn, matchings_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    matchings.emplace_back(
        cista::copy_from_potentially_unaligned<nlocation_key_t>(entry->first),
        string{entry->second});
    entry = cur.get(lmdb::cursor_op::NEXT);
  }
  cur.reset();
  return matchings;
}

hash_map<nlocation_key_t, platform> database::get_loc_to_pf_matchings() {
  auto loc_pf_matchings = hash_map<nlocation_key_t, platform>{};

  for (auto& [location, osm_key] : get_matchings()) {
    auto const pf = get_platform(osm_key);

    if (pf.has_value()) {
      loc_pf_matchings.insert(
          std::pair<nlocation_key_t, platform>(location, pf.value()));
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

/**
 * merge and update: transfer_request_keys in db
 */
std::vector<std::size_t> database::update_transfer_requests_keys(
    transfer_requests_keys const& treqs_k) {
  auto updated_indices = std::vector<std::size_t>{};
  auto treq_chashing = cista::hashing<transfer_request_keys>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto transreqs_db = transreqs_dbi(txn);

  for (auto [idx, treq] : utl::enumerate(treqs_k)) {
    auto treq_key = to_key(treq);

    if (auto const r = txn.get(transreqs_db, treq_key); !r.has_value()) {
      continue;  // transfer request not in db
    }

    auto entry = txn.get(transreqs_db, treq_key);
    auto treq_from_db =
        cista::copy_from_potentially_unaligned<transfer_request_keys>(
            entry.value());
    auto merged = merge(treq_from_db, treq);

    // update entry only in case of changes
    if (treq_chashing(treq_from_db) == treq_chashing(merged)) {
      continue;
    }

    auto const serialized_treq = cista::serialize(merged);
    if (txn.del(transreqs_db, treq_key)) {
      txn.put(transreqs_db, treq_key, view(serialized_treq));
    }

    updated_indices.emplace_back(idx);
  }

  txn.commit();
  return updated_indices;
}

transfer_requests_keys database::get_transfer_requests_keys(
    set<profile_key_t> const& ppr_profile_names) {
  auto treqs_k = transfer_requests_keys{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto transreqs_db = transreqs_dbi(txn);
  auto cur = lmdb::cursor{txn, transreqs_db};
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    auto const db_treq_k =
        cista::copy_from_potentially_unaligned<transfer_request_keys>(
            entry->second);

    // extract only transfer_requests with requested profiles
    if (ppr_profile_names.count(db_treq_k.profile_) == 1) {
      treqs_k.emplace_back(db_treq_k);
    }

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

/**
 * merge and update: transfer_results in db
 */
std::vector<std::size_t> database::update_transfer_results(
    transfer_results const& trs) {
  auto updated_indices = std::vector<std::size_t>{};
  auto tres_chashing = cista::hashing<transfer_result>{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_};
  auto transfers_db = transfers_dbi(txn);

  for (auto const& [idx, tres] : utl::enumerate(trs)) {
    auto const tres_key = to_key(tres);

    if (auto const r = txn.get(transfers_db, tres_key); !r.has_value()) {
      continue;  // transfer not in db
    }

    auto entry = txn.get(transfers_db, tres_key);
    auto tres_from_db =
        cista::copy_from_potentially_unaligned<transfer_result>(entry.value());
    auto merged = merge(tres_from_db, tres);

    // update entry only in case of changes
    if (tres_chashing(tres_from_db) == tres_chashing(merged)) {
      continue;
    }

    auto const serialized_tr = cista::serialize(merged);
    if (txn.del(transfers_db, tres_key)) {
      txn.put(transfers_db, tres_key, view(serialized_tr));
    }

    updated_indices.emplace_back(idx);
  }

  txn.commit();
  return updated_indices;
}

transfer_results database::get_transfer_results(
    set<profile_key_t> const& ppr_profile_names) {
  auto trs = transfer_results{};

  auto lock = std::lock_guard{mutex_};
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto transfers_db = transfers_dbi(txn);
  auto cur = lmdb::cursor(txn, transfers_db);
  auto entry = cur.get(lmdb::cursor_op::FIRST);

  while (entry.has_value()) {
    auto const db_tr =
        cista::copy_from_potentially_unaligned<transfer_result>(entry->second);

    // extract only transfer_results with requested profiles
    if (ppr_profile_names.count(db_tr.profile_) == 1) {
      trs.emplace_back(db_tr);
    }

    entry = cur.get(lmdb::cursor_op::NEXT);
  }

  cur.reset();
  return trs;
}

lmdb::txn::dbi database::profiles_dbi(lmdb::txn& txn, lmdb::dbi_flags flags) {
  return txn.dbi_open(kProfilesDB, flags);
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
