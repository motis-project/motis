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

}  // namespace motis::odm