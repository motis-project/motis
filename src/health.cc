#include "motis/health.h"

namespace motis {

bool rt_updated(metrics_registry const& m) {
  return m.last_update_rt_.Value() > 0.0;
}

bool gbfs_updated(metrics_registry const& m) {
  return m.last_update_gbfs_.Value() > 0.0;
}

bool rt_healthy(config const& c, metrics_registry const& m) {
  return !c.requires_rt_timetable_updates() || rt_updated(m);
}

bool gbfs_healthy(config const& c, metrics_registry const& m) {
  return !c.has_gbfs_feeds() || gbfs_updated(m);
}

bool is_healthy(config const& c, metrics_registry const& m) {
  return rt_healthy(c, m) && gbfs_healthy(c, m);
}

}  // namespace motis
