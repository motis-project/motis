#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lmdb/lmdb.hpp"

#include "tiles/db/clear_database.h"
#include "tiles/db/pack_file.h"
#include "tiles/db/tile_database.h"

#include "utl/verify.h"

#include "motis/path/error.h"

namespace motis::path {

struct path_database {
  path_database(std::string const& path, bool read_only)
      : read_only_{read_only} {
    env_.set_mapsize(static_cast<mdb_size_t>(32) * 1024 * 1024 * 1024);
    env_.set_maxdbs(8);

    auto open_flags = lmdb::env_open_flags::NOSUBDIR;
    if (read_only) {
      open_flags = open_flags | lmdb::env_open_flags::RDONLY;
    }

    env_.open(path.c_str(), open_flags);
    db_handle_ = std::make_unique<tiles::tile_db_handle>(env_);
    pack_handle_ = std::make_unique<tiles::pack_handle>(path.c_str());

    auto txn = db_handle_->make_txn();
    data_dbi(txn, lmdb::dbi_flags::CREATE);
    txn.commit();
  }

  static lmdb::txn::dbi data_dbi(
      lmdb::txn& txn, lmdb::dbi_flags flags = lmdb::dbi_flags::NONE) {
    return txn.dbi_open("motis-path-data", flags);
  }

  std::string get(std::string const& k) const {
    auto ret = try_get(k);
    if (!ret) {
      throw std::system_error(error::not_found);
    }
    return *ret;
  }

  std::optional<std::string> try_get(std::string const& k) const {
    auto txn = db_handle_->make_txn();
    auto db = data_dbi(txn);

    auto ret = txn.get(db, k);
    if (ret) {
      return std::optional<std::string>{*ret};
    }
    return std::nullopt;
  }

  lmdb::env env_;
  std::unique_ptr<tiles::tile_db_handle> db_handle_;
  std::unique_ptr<tiles::pack_handle> pack_handle_;

  bool read_only_;
};

inline std::unique_ptr<path_database> make_path_database(
    std::string const& fname, bool read_only, bool truncate = false) {
  utl::verify(!(read_only && truncate),
              "make_path_database: either truncate or read_only");
  auto db = std::make_unique<path_database>(fname, read_only);

  if (truncate) {
    auto txn = db->db_handle_->make_txn();

    auto data_dbi = db->data_dbi(txn);
    txn.dbi_clear(data_dbi);

    tiles::clear_database(*db->db_handle_, txn);

    txn.commit();

    db->pack_handle_->resize(0);
  }

  return db;
}

}  // namespace motis::path
