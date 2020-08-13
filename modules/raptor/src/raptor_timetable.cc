#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

// overload invalid for station id,
// since we have 32b and 24b station ids, which must be comparable
template <>
constexpr auto invalid<station_id> = -1;

}  // namespace motis::raptor