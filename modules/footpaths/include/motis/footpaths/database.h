#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "geo/latlng.h"

#include "lmdb/lmdb.hpp"

#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

#include "nigiri/keys.h"

namespace motis::footpaths {

struct database {
  explicit database(std::string const& path, std::size_t const max_size);

  // profiles
  std::vector<std::size_t> put_profiles(std::vector<string> const&);
  hash_map<string, key8_t> get_profile_keys();

  // platforms
  std::vector<std::size_t> put_platforms(platforms&);
  platforms get_platforms();

  // matchings
  std::vector<std::size_t> put_matching_results(matching_results const&);
  hash_map<key64_t, platform> get_loc_to_pf_matchings();

  // transfer requests
  std::vector<std::size_t> put_transfer_requests_keys(
      transfer_requests_keys const&);
  std::vector<std::size_t> update_transfer_requests_keys(
      transfer_requests_keys const&);
  transfer_requests_keys get_transfer_requests_keys(set<key8_t> const&);

  // transfer results
  std::vector<std::size_t> put_transfer_results(transfer_results const&);
  std::vector<std::size_t> update_transfer_results(transfer_results const&);
  transfer_results get_transfer_results(set<key8_t> const&);

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

  std::vector<std::pair<key64_t, string>> get_matchings();
  std::optional<platform> get_platform(string const& /* osm_key */);

  lmdb::env mutable env_;
  std::mutex mutex_;
  key8_t highest_profile_id_{};
};

}  // namespace motis::footpaths
