#pragma once

#include <optional>
#include <string>

#include "lmdb/lmdb.hpp"

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/parking/dbschema/FootEdges_generated.h"

namespace motis::parking {

using persistable_foot_edges = typed_flatbuffer<FootEdges>;

struct database {
  explicit database(std::string const& path, std::size_t max_size,
                    bool read_only);

  void put(persistable_foot_edges const& fe);
  std::optional<persistable_foot_edges> get(
      int32_t parking_id, std::string const& search_profile) const;

private:
  lmdb::env mutable env_;
  bool open_;
};

}  // namespace motis::parking
