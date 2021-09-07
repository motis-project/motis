#pragma once

#include "motis/protocol/RoutingResponse_generated.h"

namespace motis::routing::eval {

inline uint64_t get_stat(routing::RoutingResponse const* r,
                         char const* category, char const* name) {
  auto const& cat = r->statistics()->LookupByKey(category);
  if (cat == nullptr) {
    return 0;
  }
  auto const& entry = cat->entries()->LookupByKey(name);
  return entry == nullptr ? 0 : entry->value();
}

}  // namespace motis::routing::eval