#pragma once

#include "motis/fwd.h"

namespace motis {

void adr_extend_tt(nigiri::timetable const&,
                   adr::area_database const&,
                   adr::typeahead&);

}  // namespace motis