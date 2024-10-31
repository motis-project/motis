#include "motis/mode_to_profile.h"

#include "utl/verify.h"

namespace motis {

osr::search_profile to_profile(api::ModeEnum const m, bool const wheelchair) {
  switch (m) {
    case api::ModeEnum::WALK:
      return wheelchair ? osr::search_profile::kWheelchair
                        : osr::search_profile::kFoot;
    case api::ModeEnum::BIKE: return osr::search_profile::kBike;
    case api::ModeEnum::CAR: return osr::search_profile::kCar;
    case api::ModeEnum::BIKE_RENTAL: return osr::search_profile::kBikeSharing;
    default: throw utl::fail("unsupported mode");
  }
}

}  // namespace motis