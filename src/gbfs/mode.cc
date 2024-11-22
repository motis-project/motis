#include "motis/gbfs/mode.h"

#include "motis/constants.h"

namespace motis::gbfs {

api::ModeEnum get_gbfs_mode(vehicle_form_factor const ff) {
  switch (ff) {
    case gbfs::vehicle_form_factor::kBicycle:
    case gbfs::vehicle_form_factor::kCargoBicycle:
      return api::ModeEnum::BIKE_RENTAL;
    case gbfs::vehicle_form_factor::kScooterStanding:
    case gbfs::vehicle_form_factor::kScooterSeated:
      return api::ModeEnum::SCOOTER_RENTAL;
    case gbfs::vehicle_form_factor::kCar: return api::ModeEnum::CAR_RENTAL;
    default: return api::ModeEnum::BIKE_RENTAL;
  }
}

api::ModeEnum get_gbfs_mode(gbfs_data const& gbfs, gbfs_segment_ref const ref) {
  return get_gbfs_mode(gbfs.providers_.at(ref.provider_)
                           ->segments_.at(ref.segment_)
                           .form_factor_);
}

api::ModeEnum get_gbfs_mode(gbfs_routing_data const& gbfs_rd,
                            nigiri::transport_mode_id_t const id) {
  return get_gbfs_mode(*gbfs_rd.data_, gbfs_rd.get_segment_ref(id));
}

bool form_factor_matches(api::ModeEnum const m,
                         gbfs::vehicle_form_factor const ff) {
  switch (m) {
    case api::ModeEnum::BIKE_RENTAL:
      return ff == gbfs::vehicle_form_factor::kBicycle ||
             ff == gbfs::vehicle_form_factor::kCargoBicycle;
    case api::ModeEnum::SCOOTER_RENTAL:
      return ff == gbfs::vehicle_form_factor::kScooterStanding ||
             ff == gbfs::vehicle_form_factor::kScooterSeated;
    default: return false;
  }
}

}  // namespace motis::gbfs
