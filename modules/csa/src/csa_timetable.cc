#include "motis/csa/csa_timetable.h"

#include "motis/core/schedule/station.h"

namespace motis::csa {

csa_station::csa_station(station const* station_ptr)
    : id_(station_ptr->index_),
      transfer_time_(station_ptr->transfer_time_),
      footpaths_({{id_, id_, transfer_time_}}),
      incoming_footpaths_({{id_, id_, transfer_time_}}),
      station_ptr_(station_ptr) {}

}  // namespace motis::csa