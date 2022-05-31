#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

// overload invalid for station id,
// since we have 32b and 24b station ids, which must be comparable
template <>
constexpr auto invalid<stop_id> = -1;

}  // namespace motis::mcraptor