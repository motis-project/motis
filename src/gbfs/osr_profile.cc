#include "motis/gbfs/osr_profile.h"

namespace motis::gbfs {

osr::search_profile get_osr_profile(vehicle_form_factor const& ff) {
  return ff == vehicle_form_factor::kCar ? osr::search_profile::kCarSharing
                                         : osr::search_profile::kBikeSharing;
}

}  // namespace motis::gbfs
