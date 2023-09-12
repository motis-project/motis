#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "geo/latlng.h"

#include "lmdb/lmdb.hpp"

#include "motis/transfers/matching.h"
#include "motis/transfers/platform/platform.h"
#include "motis/transfers/transfer/transfer_request.h"
#include "motis/transfers/transfer/transfer_result.h"
#include "motis/transfers/types.h"

namespace motis::transfers {

struct database {
  explicit database(std::filesystem::path const& db_file_path,
                    std::size_t const db_max_size);

  // profiles
  std::vector<std::size_t> put_profiles(std::vector<string> const&);
  hash_map<string, profile_key_t> get_profile_keys();
  hash_map<profile_key_t, string> get_profile_key_to_name();

  // platforms
  std::vector<std::size_t> put_platforms(platforms&);
  platforms get_platforms();

  // matchings
  std::vector<std::size_t> put_matching_results(matching_results const&);
  hash_map<nlocation_key_t, platform> get_loc_to_pf_matchings();

  // transfer requests
  std::vector<std::size_t> put_transfer_requests_keys(
      transfer_requests_keys const&);
  std::vector<std::size_t> update_transfer_requests_keys(
      transfer_requests_keys const&);
  transfer_requests_keys get_transfer_requests_keys(set<profile_key_t> const&);

  // transfer results
  std::vector<std::size_t> put_transfer_results(transfer_results const&);
  std::vector<std::size_t> update_transfer_results(transfer_results const&);
  transfer_results get_transfer_results(set<profile_key_t> const&);

private:
  static lmdb::txn::dbi profiles_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi platforms_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi matchings_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi transreqs_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi transfers_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  void init();

  std::vector<std::pair<nlocation_key_t, string>> get_matchings();
  std::optional<platform> get_platform(string const& /* osm_key */);

  lmdb::env mutable env_;
  std::mutex mutex_;
  profile_key_t highest_profile_id_{};
};

}  // namespace motis::transfers
