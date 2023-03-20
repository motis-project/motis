#pragma once

#include <string>
#include <vector>

#include "motis/core/journey/journey.h"

namespace nigiri {
struct timetable;
namespace routing {
struct journey;
}
}  // namespace nigiri

namespace motis::nigiri {

motis::journey nigiri_to_motis_journey(::nigiri::timetable const&,
                                       std::vector<std::string> const&,
                                       ::nigiri::routing::journey const&);

}  // namespace motis::nigiri
