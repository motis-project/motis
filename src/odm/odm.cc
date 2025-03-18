#include "motis/odm/odm.h"

#include <ranges>

#include "nigiri/routing/journey.h"

#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;
namespace nr = nigiri::routing;

bool is_odm_leg(nr::journey::leg const& l) {
  return std::holds_alternative<nr::offset>(l.uses_) &&
         std::get<nr::offset>(l.uses_).transport_mode_id_ ==
             kOdmTransportModeId;
}

bool uses_odm(nr::journey const& j) { return utl::any_of(j.legs_, is_odm_leg); }

bool is_pure_pt(nr::journey const& j) { return !uses_odm(j); };

bool is_direct_odm(nr::journey const& j) {
  return j.legs_.size() == 1U && uses_odm(j);
}

n::duration_t odm_time(nr::journey::leg const& l) {
  return is_odm_leg(l) ? std::get<nr::offset>(l.uses_).duration()
                       : n::duration_t{0};
}

n::duration_t odm_time(nr::journey const& j) {
  return std::transform_reduce(begin(j.legs_), end(j.legs_), n::duration_t{0},
                               std::plus{},
                               [](auto const& l) { return odm_time(l); });
}

}  // namespace motis::odm