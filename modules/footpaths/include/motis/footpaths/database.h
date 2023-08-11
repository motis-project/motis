#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfer_updates.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

inline std::string to_key(platform const& pf) {
  return fmt::format("{}:{}", std::to_string(pf.info_.osm_id_),
                     get_osm_str_type(pf.info_.osm_type_));
}

inline std::string to_key(geo::latlng const& pos) {
  return fmt::format("{}:{}", std::to_string(pos.lat_),
                     std::to_string(pos.lng_));
}

inline std::string to_key(transfer_result const& tr) {
  return fmt::format("{}::{}::{}", to_key(tr.from_), to_key(tr.to_),
                     tr.profile_);
}

struct database {
  explicit database(std::string const& path, std::size_t const max_size);

  std::vector<std::size_t> put_platforms(std::vector<platform>&);
  std::vector<platform> get_platforms();
  std::vector<platform> get_matched_platforms();

  std::vector<std::size_t> put_matching_results(matching_results const&);

  hash_map<std::string, platform> get_loc_to_pf_matchings();

  std::vector<std::size_t> put_transfer_results(transfer_results const&);
  hash_map<std::string, transfer_result> get_trs_with_key();

private:
  static lmdb::txn::dbi platforms_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi matchings_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi transfers_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  void init();

  std::vector<std::pair<std::string, std::string>> get_matchings();
  std::optional<platform> get_platform(std::string const& /* osm_key */);

  lmdb::env mutable env_;
  std::mutex mutex_;
  std::int32_t highest_platform_id_{};
};

}  // namespace motis::footpaths