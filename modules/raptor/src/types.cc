#include "motis/raptor/types.h"

namespace motis::raptor {
// overload invalid for station id,
// since we have 32b and 24b station ids, which must be comparable
template <>
constexpr auto invalid<stop_id> = -1;
}