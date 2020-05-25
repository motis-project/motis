#pragma once

#include <map>
#include <string>

#include "motis/loader/gtfs/trip.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

void read_stop_times(loaded_file const&, trip_map&, stop_map const&,
                     bool shorten_stop_ids);

}  // namespace motis::loader::gtfs
