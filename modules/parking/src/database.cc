#include "motis/parking/database.h"

#include "boost/filesystem.hpp"

#include "fmt/core.h"

#include "motis/core/common/logging.h"

namespace fs = boost::filesystem;
using namespace motis::logging;

namespace motis::parking {

database::database(std::string const& path, std::size_t const max_size,
                   bool read_only) {
  env_.set_maxdbs(1);
  env_.set_mapsize(max_size);
  auto flags = lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOSYNC;
  if (read_only) {
    flags = flags | lmdb::env_open_flags::NOLOCK | lmdb::env_open_flags::NOTLS;
  }
  if ((path != "-" && fs::exists(path)) || !read_only) {
    env_.open(path.c_str(), flags);
    open_ = true;
  } else {
    open_ = false;
    LOG(warn) << "Database not found: " << path;
  }
}

inline std::string get_db_key(int32_t parking_id, std::string const& profile) {
  return fmt::format("{}:{}", parking_id, profile);
}

void database::put(const persistable_foot_edges& fe) {
  auto txn = lmdb::txn{env_};
  auto db = txn.dbi_open();
  auto const key =
      get_db_key(fe.get()->parking_id(), fe.get()->search_profile()->str());
  txn.put(db, key, fe.to_string());
  txn.commit();
}

std::optional<persistable_foot_edges> database::get(
    int32_t parking_id, std::string const& search_profile) const {
  if (!open_) {
    return {};
  }
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto db = txn.dbi_open();
  auto const key = get_db_key(parking_id, search_profile);
  if (auto const r = txn.get(db, key); !r.has_value()) {
    return {};
  } else {
    return {persistable_foot_edges(*r)};
  }
}

}  // namespace motis::parking
