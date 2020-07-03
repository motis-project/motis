#pragma once

#include <string>
#include <vector>

#include "motis/hash_map.h"

#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/schedule/connection.h"

#include "motis/path/fbs/PathIndex_generated.h"

namespace motis::path {

struct path_index {
  using index_t = typed_flatbuffer<PathIndex>;

  struct seq_key {
    std::vector<std::string> station_ids_;
    service_class clasz_{service_class::OTHER};
  };

  struct seq_info {
    std::vector<std::string> station_ids_;
    std::vector<service_class> classes_;
  };

  path_index() = default;
  explicit path_index(std::string const& s);
  explicit path_index(index_t);

  size_t find(seq_key const& k) const;

  struct segment_info {
    std::string from_, to_;
    std::vector<service_class> classes_;
  };

  std::vector<segment_info> get_segments(size_t ref) const;

  mcd::hash_map<seq_key, size_t> seq_map_;

  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> tile_features_;
  std::vector<seq_info> seq_keys_;
};

}  // namespace motis::path
