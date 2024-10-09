#pragma once

#include "nigiri/rt/run.h"
#include "nigiri/types.h"

#include "motis/core/journey/extern_trip.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

struct tag_lookup;

::nigiri::rt::run resolve_run(tag_lookup const&, ::nigiri::timetable const&,
                              extern_trip const&);

}  // namespace motis::nigiri