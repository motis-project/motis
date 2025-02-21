#include "motis/timetable/modes_to_clasz_mask.h"

namespace n = nigiri;

namespace motis {

n::routing::clasz_mask_t to_clasz_mask(std::vector<api::ModeEnum> const& mode) {
  auto mask = n::routing::clasz_mask_t{0U};
  auto const allow = [&](n::clasz const c) {
    mask |= (1U << static_cast<std::underlying_type_t<n::clasz>>(c));
  };
  for (auto const& m : mode) {
    switch (m) {
      case api::ModeEnum::TRANSIT:
        mask = n::routing::all_clasz_allowed();
        return mask;
      case api::ModeEnum::TRAM: allow(n::clasz::kTram); break;
      case api::ModeEnum::SUBWAY: allow(n::clasz::kSubway); break;
      case api::ModeEnum::FERRY: allow(n::clasz::kShip); break;
      case api::ModeEnum::AIRPLANE: allow(n::clasz::kAir); break;
      case api::ModeEnum::BUS: allow(n::clasz::kBus); break;
      case api::ModeEnum::COACH: allow(n::clasz::kCoach); break;
      case api::ModeEnum::RAIL:
        allow(n::clasz::kHighSpeed);
        allow(n::clasz::kLongDistance);
        allow(n::clasz::kNight);
        allow(n::clasz::kRegional);
        allow(n::clasz::kRegionalFast);
        allow(n::clasz::kMetro);
        allow(n::clasz::kSubway);
        break;
      case api::ModeEnum::HIGHSPEED_RAIL: allow(n::clasz::kHighSpeed); break;
      case api::ModeEnum::LONG_DISTANCE: allow(n::clasz::kLongDistance); break;
      case api::ModeEnum::NIGHT_RAIL: allow(n::clasz::kNight); break;
      case api::ModeEnum::REGIONAL_FAST_RAIL:
        allow(n::clasz::kRegionalFast);
        break;
      case api::ModeEnum::REGIONAL_RAIL: allow(n::clasz::kRegional); break;
      case api::ModeEnum::METRO: allow(n::clasz::kMetro); break;
      default: continue;
    }
  }
  return mask;
}

}  // namespace motis