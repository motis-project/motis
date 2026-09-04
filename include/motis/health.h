#pragma once

#include "motis/config.h"
#include "motis/metrics_registry.h"

namespace motis {

// True if no rt/elevator feed is configured, or one has updated at least
// once. Sticky, since last_update_rt_ is never reset once set.
bool rt_healthy(config const&, metrics_registry const&);

bool gbfs_healthy(config const&, metrics_registry const&);

bool is_healthy(config const&, metrics_registry const&);

}  // namespace motis
