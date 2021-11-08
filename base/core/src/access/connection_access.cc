#include "motis/core/access/connection_access.h"

#include <algorithm>
#include <iterator>

namespace motis::access {

connection_info const& get_connection_info(schedule const& sched,
                                           light_connection const& lcon,
                                           trip_info const* trp) {
  auto const& trips = *sched.merged_trips_[lcon.trips_];
  if (trips.size() == 1) {
    return *lcon.full_con_->con_info_;
  }

  auto const it = std::find(begin(trips), end(trips), trp);
  assert(it != end(trips));
  auto const pos = std::distance(begin(trips), it);

  auto info = lcon.full_con_->con_info_;
  for (int i = 0; i < pos; ++i) {
    info = info->merged_with_;
  }
  return *info;
}

}  // namespace motis::access
