#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis {

api::ModeEnum to_mode(n::clasz const c) {
  switch (c) {
    case n::clasz::kAir: return api::ModeEnum::AIRPLANE;
    case n::clasz::kHighSpeed: return api::ModeEnum::HIGHSPEED_RAIL;
    case n::clasz::kLongDistance: return api::ModeEnum::LONG_DISTANCE;
    case n::clasz::kCoach: return api::ModeEnum::COACH;
    case n::clasz::kNight: return api::ModeEnum::NIGHT_RAIL;
    case n::clasz::kRegionalFast: return api::ModeEnum::REGIONAL_FAST_RAIL;
    case n::clasz::kRegional: return api::ModeEnum::REGIONAL_RAIL;
    case n::clasz::kMetro: return api::ModeEnum::METRO;
    case n::clasz::kSubway: return api::ModeEnum::SUBWAY;
    case n::clasz::kTram: return api::ModeEnum::TRAM;
    case n::clasz::kBus: return api::ModeEnum::BUS;
    case n::clasz::kShip: return api::ModeEnum::FERRY;
    case n::clasz::kCableCar: return api::ModeEnum::CABLE_CAR;
    case n::clasz::kFunicular: return api::ModeEnum::FUNICULAR;
    case n::clasz::kAreaLift: return api::ModeEnum::AREAL_LIFT;
    case n::clasz::kOther: return api::ModeEnum::OTHER;
    case n::clasz::kNumClasses:;
  }
  std::unreachable();
}

}  // namespace motis