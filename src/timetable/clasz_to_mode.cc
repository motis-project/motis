#include "motis/timetable/clasz_to_mode.h"

#include "utl/for_each_bit_set.h"

namespace n = nigiri;

namespace motis {

api::ModeEnum to_mode(n::clasz const c, unsigned const api_version) {
  switch (c) {
    case n::clasz::kAir: return api::ModeEnum::AIRPLANE;
    case n::clasz::kHighSpeed: return api::ModeEnum::HIGHSPEED_RAIL;
    case n::clasz::kLongDistance: return api::ModeEnum::LONG_DISTANCE;
    case n::clasz::kCoach: return api::ModeEnum::COACH;
    case n::clasz::kNight: return api::ModeEnum::NIGHT_RAIL;
    case n::clasz::kRideSharing: return api::ModeEnum::RIDE_SHARING;
    case n::clasz::kRegional: return api::ModeEnum::REGIONAL_RAIL;
    case n::clasz::kSuburban:
      return api_version < 5 ? api::ModeEnum::METRO : api::ModeEnum::SUBURBAN;
    case n::clasz::kSubway: return api::ModeEnum::SUBWAY;
    case n::clasz::kTram: return api::ModeEnum::TRAM;
    case n::clasz::kBus: return api::ModeEnum::BUS;
    case n::clasz::kShip: return api::ModeEnum::FERRY;
    case n::clasz::kODM: return api::ModeEnum::ODM;
    case n::clasz::kFunicular: return api::ModeEnum::FUNICULAR;
    case n::clasz::kAerialLift:
      return api_version < 5 ? api::ModeEnum::AREAL_LIFT
                             : api::ModeEnum::AERIAL_LIFT;
    case n::clasz::kOther: return api::ModeEnum::OTHER;
    case n::clasz::kNumClasses:;
  }
  std::unreachable();
}

std::vector<api::ModeEnum> to_modes(nigiri::routing::clasz_mask_t const mask,
                                    unsigned api_version) {
  auto modes = std::vector<api::ModeEnum>{};
  utl::for_each_set_bit(mask, [&](auto const i) {
    modes.emplace_back(to_mode(static_cast<n::clasz>(i), api_version));
  });
  return modes;
}

}  // namespace motis