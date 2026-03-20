#include "motis/display_filter.h"

#include "utl/verify.h"

namespace motis {

int min_zoom_level(n::clasz const clasz, float const distance) {
  switch (clasz) {
    // long distance
    case n::clasz::kAir:
    case n::clasz::kCoach:
      if (distance < 50'000.F) {
        return 8;  // typically long distance, maybe also quite short
      }
      [[fallthrough]];
    case n::clasz::kHighSpeed:
    case n::clasz::kLongDistance:
    case n::clasz::kNight: return 4;
    case n::clasz::kRideSharing:
    case n::clasz::kRegional: return 7;

    // regional distance
    case n::clasz::kSuburban: return 8;

    // metro distance
    case n::clasz::kSubway: return 9;

    // short distance
    case n::clasz::kTram:
    case n::clasz::kBus: return distance > 10'000.F ? 9 : 10;

    // ship can be anything
    case n::clasz::kShip:
      if (distance > 100'000.F) {
        return 5;
      } else if (distance > 10'000.F) {
        return 8;
      } else {
        return 10;
      }

    case n::clasz::kODM:
    case n::clasz::kFunicular:
    case n::clasz::kAerialLift:
    case n::clasz::kOther: return 11;

    default: throw utl::fail("unknown n::clasz {}", static_cast<int>(clasz));
  }
}

bool should_display(n::clasz const clasz,
                    int const zoom_level,
                    float const distance) {
  return zoom_level >= min_zoom_level(clasz, distance);
}

} // namespace motis
