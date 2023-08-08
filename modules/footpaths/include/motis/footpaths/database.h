#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"

namespace motis::footpaths {

struct database {
  explicit database(std::string const& path, std::size_t const max_size);

  std::vector<std::size_t> put_platforms(std::vector<platform>&);
  std::vector<platform> get_platforms();

  std::vector<std::size_t> put_matching_results(std::vector<matching_result>&);

private:
  static lmdb::txn::dbi platforms_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);
  static lmdb::txn::dbi matchings_dbi(
      lmdb::txn&, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  void init();

  lmdb::env mutable env_;
  std::mutex mutex_;
  std::int32_t highest_platform_id_{};
};

}  // namespace motis::footpaths