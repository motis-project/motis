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
#include "motis/footpaths/types.h"

namespace motis::footpaths {

inline std::string to_key(platform const& pf) {
  return fmt::format("{}:{}", get_osm_str_type(pf.info_.osm_type_),
                     pf.info_.osm_id_);
}

inline std::string to_key(geo::latlng const& pos) {
  return fmt::format("{}:{}", pos.lat_, pos.lng_);
}

struct database {
  explicit database(std::string const& path, std::size_t const max_size);

  std::vector<std::size_t> put_platforms(std::vector<platform>&);
  std::vector<platform> get_platforms();
  std::vector<platform> get_matched_platforms();

  std::vector<std::size_t> put_matching_results(std::vector<matching_result>&);

  hash_map<std::string, platform> get_loc_to_pf_matchings();

private:
  static lmdb::txn::dbi platforms_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi matchings_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  void init();

  std::vector<std::pair<std::string, std::string>> get_matchings();
  std::optional<platform> get_platform(std::string const& /* osm_key */);

  lmdb::env mutable env_;
  std::mutex mutex_;
  std::int32_t highest_platform_id_{};
};

}  // namespace motis::footpaths