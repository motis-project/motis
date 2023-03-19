#include "motis/paxmon/access/journeys.h"

namespace motis::paxmon {

fws_compact_journey add_compact_journey(universe& uv,
                                        compact_journey const& cj) {
  return to_fws_compact_journey(uv.passenger_groups_.compact_journey_legs_,
                                uv.passenger_groups_.final_footpaths_, cj);
}

}  // namespace motis::paxmon
