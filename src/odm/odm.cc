#include "motis/odm/odm.h"

#include "nigiri/routing/journey.h"

#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;

bool is_odm_leg(nigiri::routing::journey::leg const& l) {
  return std::holds_alternative<n::routing::offset>(l.uses_) &&
         std::get<n::routing::offset>(l.uses_).transport_mode_id_ ==
             kOdmTransportModeId;
}

n::duration_t odm_time(n::routing::journey const& j) {
  return j.legs_.empty() ? n::duration_t{0}
         : is_odm_leg(j.legs_.front())
             ? std::get<n::routing::offset>(j.legs_.front().uses_).duration()
             : n::duration_t{0} +
                   ((j.legs_.size() > 1 && is_odm_leg(j.legs_.back()))
                        ? std::get<n::routing::offset>(j.legs_.back().uses_)
                              .duration()
                        : n::duration_t{0});
}

}  // namespace motis::odm