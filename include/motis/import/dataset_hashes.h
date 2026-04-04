#pragma once

#include "motis/fwd.h"
#include "motis/hashes.h"

namespace motis {

struct dataset_hashes {
  explicit dataset_hashes(config const&);

  meta_entry_t osm_;
  meta_entry_t tt_;
  meta_entry_t elevation_;
  meta_entry_t tiles_;
};

}  // namespace motis