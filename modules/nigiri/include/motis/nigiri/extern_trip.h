#pragma once

#include "nigiri/types.h"

#include "motis/core/journey/extern_trip.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;

extern_trip nigiri_trip_to_extern_trip(tag_lookup const&,
                                       ::nigiri::timetable const&,
                                       ::nigiri::trip_idx_t,
                                       ::nigiri::transport);

}  // namespace motis::nigiri