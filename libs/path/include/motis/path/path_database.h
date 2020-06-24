#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "tiles/db/pack_file.h"
#include "tiles/db/tile_database.h"

namespace motis::path {

struct path_database {
  path_database(std::string const& fname, bool read_only);

  static lmdb::txn::dbi data_dbi(lmdb::txn& txn,
                                 lmdb::dbi_flags flags = lmdb::dbi_flags::NONE);

  std::string get(std::string const& k) const;
  std::optional<std::string> try_get(std::string const& k) const;

  lmdb::env env_;
  std::unique_ptr<tiles::tile_db_handle> db_handle_;
  std::unique_ptr<tiles::pack_handle> pack_handle_;

  bool read_only_;
};

std::unique_ptr<path_database> make_path_database(std::string const& fname,
                                                  bool read_only,
                                                  bool truncate = false);

}  // namespace motis::path
