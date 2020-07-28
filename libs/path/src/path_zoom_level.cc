#include "motis/path/path_zoom_level.h"

#include "utl/verify.h"

namespace motis::path {

int min_zoom_level(service_class const clasz, float const distance) {
  switch (clasz) {
    // long distance
    case service_class::AIR:
    case service_class::COACH:
      if (distance < 50'000.F) {
        return 8;  // typically long distance, maybe also quite short
      }
      [[fallthrough]];
    case service_class::ICE:
    case service_class::IC:
    case service_class::N: return 4;

    // regional distance
    case service_class::RE:
    case service_class::RB:
    case service_class::S: return 5;

    // metro distance
    case service_class::U: return 6;

    // short distance
    case service_class::STR:
    case service_class::BUS: return distance > 10'000.F ? 10 : 11;

    // ship can be anything
    case service_class::SHIP:
      if (distance > 100'000.F) {
        return 5;
      } else if (distance > 10'000.F) {
        return 8;
      } else {
        return 10;
      }

    case service_class::OTHER: return 11;
    default:
      throw utl::fail("unknown service_class {}",
                      static_cast<service_class_t>(clasz));
  }
}

bool should_display(service_class const clasz, int const zoom_level,
                    float const distance) {
  return zoom_level >= min_zoom_level(clasz, distance);
}

}  // namespace motis::path
