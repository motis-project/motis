#pragma once

#include "motis/core/journey/journey.h"
#include "motis/nigiri/tag_lookup.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
namespace routing {
struct journey;
}
}  // namespace nigiri

namespace motis::nigiri {

motis::journey nigiri_to_motis_journey(::nigiri::timetable const&,
                                       ::nigiri::rt_timetable const*,
                                       tag_lookup const&,
                                       ::nigiri::routing::journey const&);

}  // namespace motis::nigiri
