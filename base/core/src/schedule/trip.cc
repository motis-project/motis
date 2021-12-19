#include "motis/core/schedule/trip.h"

#include "motis/core/common/date_time_util.h"

namespace motis {

std::ostream& operator<<(std::ostream& out, gtfs_trip_id const& id) {
  return out << "{GTFS-TRIP trip_id=" << id.trip_id_ << ", start_date="
             << (id.start_date_.has_value() ? format_unix_time(*id.start_date_)
                                            : "NO VALUE")
             << "}";
}

}  // namespace motis