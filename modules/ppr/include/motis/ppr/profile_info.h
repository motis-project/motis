#pragma once

#include <cstddef>

#include "cista/hash.h"

#include "boost/filesystem.hpp"

#include "ppr/routing/search_profile.h"

namespace motis::ppr {

struct profile_info {
  ::ppr::routing::search_profile profile_;
  boost::filesystem::path file_path_;
  cista::hash_t file_hash_{};
  std::size_t file_size_{};
};

}  // namespace motis::ppr
