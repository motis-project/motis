#include "motis/paxforecast/routing_cache.h"

using namespace motis::module;

namespace motis::paxforecast {

void routing_cache::open(const std::string& path) {
  env_.set_maxdbs(1);
  env_.set_mapsize(50ULL * 1024 * 1024 * 1024);
  env_.open(path.c_str(),
            lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOSYNC);
}

void routing_cache::put(const std::string_view& key, msg_ptr const& msg) {
  auto txn = lmdb::txn{env_};
  auto db = txn.dbi_open();
  txn.put(db, key, msg->to_string());
  txn.commit();
}

msg_ptr routing_cache::get(const std::string_view& key) {
  auto txn = lmdb::txn{env_, lmdb::txn_flags::RDONLY};
  auto db = txn.dbi_open();
  if (auto const r = txn.get(db, key); r.has_value()) {
    return make_msg(r->data(), r->size());
  } else {
    return {};
  }
}

void routing_cache::sync() {
  if (is_open()) {
    env_.force_sync();
  }
}

}  // namespace motis::paxforecast
