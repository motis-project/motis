#pragma once

#include "motis/config.h"
#include "motis/metrics_registry.h"

namespace motis {

bool rt_updated(metrics_registry const&);
bool gbfs_updated(metrics_registry const&);

bool rt_healthy(config const&, metrics_registry const&);

bool gbfs_healthy(config const&, metrics_registry const&);

bool is_healthy(config const&, metrics_registry const&);

}  // namespace motis
