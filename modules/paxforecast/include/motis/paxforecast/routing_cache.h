#pragma once

#include <string>
#include <string_view>

#include "lmdb/lmdb.hpp"

#include "motis/module/message.h"

namespace motis::paxforecast {

struct routing_cache {
  void open(std::string const& path);

  inline bool is_open() const { return env_.is_open(); }

  void put(std::string_view const& key, motis::module::msg_ptr const& msg);
  motis::module::msg_ptr get(std::string_view const& key);

  void sync();

private:
  lmdb::env env_;
};

}  // namespace motis::paxforecast
