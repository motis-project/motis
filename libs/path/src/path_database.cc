#include "motis/path/path_database.h"

#include "boost/filesystem.hpp"

#include "tiles/db/clear_database.h"

#include "utl/verify.h"

#include "motis/path/error.h"

namespace fs = boost::filesystem;

namespace motis::path {

path_database::path_database(std::string const& fname, bool const read_only,
                             size_t const max_size)
    : read_only_{read_only} {
  env_.set_mapsize(max_size);
  env_.set_maxdbs(8);

  auto open_flags = lmdb::env_open_flags::NOSUBDIR;
  if (read_only) {
    open_flags = open_flags | lmdb::env_open_flags::RDONLY;
  }

  env_.open(fname.c_str(), open_flags);
  db_handle_ = std::make_unique<tiles::tile_db_handle>(env_);
  pack_handle_ = std::make_unique<tiles::pack_handle>(fname.c_str());

  auto txn = db_handle_->make_txn();
  data_dbi(txn, lmdb::dbi_flags::CREATE);
  txn.commit();
}

lmdb::txn::dbi path_database::data_dbi(lmdb::txn& txn,
                                       lmdb::dbi_flags const flags) {
  return txn.dbi_open("motis-path-data", flags);
}

std::string path_database::get(std::string const& k) const {
  auto ret = try_get(k);
  utl::verify_ex(ret.has_value(), std::system_error{error::not_found});
  return *ret;  // NOLINT(bugprone-unchecked-optional-access)
}

std::optional<std::string> path_database::try_get(std::string const& k) const {
  auto txn = db_handle_->make_txn();
  auto db = data_dbi(txn);

  auto ret = txn.get(db, k);
  if (ret) {
    return std::optional<std::string>{*ret};
  }
  return std::nullopt;
}

std::unique_ptr<path_database> make_path_database(std::string const& fname,
                                                  bool const read_only,
                                                  bool const truncate,
                                                  size_t const max_size) {
  utl::verify(!(read_only && truncate),
              "make_path_database: either truncate or read_only");
  if (auto p = fs::path(fname); p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }
  auto db = std::make_unique<path_database>(fname, read_only, max_size);

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
