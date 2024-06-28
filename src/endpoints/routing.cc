#include "icc/endpoints/routing.h"

#include "icc/endpoints/graph.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

namespace json = boost::json;

namespace icc::ep {

// $$mode: list of comma separated modes
//
// first/last leg modes:
//   - WALK
//   - TRANSIT
//   - BICYCLE
//   - BICYCLE_RENT
//   - BICYCLE_PARK
//   - CAR
//   - CAR_PARK
//
// timetable based modes:
//   - TRAM = kTram
//   - SUBWAY = kSubway | kMetro
//   - RAIL = kHighspeed | kLongDistance | kNight | kRegional | kRegionalFast
//   - BUS = kBus | kCoach
//   - FERRY = kShip
//   - AIRPLANE = kAir
//   - not supported: GONDOLA
//   - not supported: FUNICULAR
//   - not supported: CABLE_CAR

// Query Parameters (* = from OTP)
//
// * maxTransfers = INTEGER | maximum number of allowed transfers
// * maxHours = DOUBLE | maximum number of hours
// * mode = STRING | $$mode
// * numItineraries = INTEGER | the minimum number of itineraries to find
// * pageCursor = STRING | "LATER|$$timestamp" or "EARLIER|$$timestamp"
// * searchWindow = INTEGER | length of the search window in seconds
// * timetableView = BOOLEAN | true=pretrip, false=ontrip
// * date = departure date ($arriveBy=false) / arrival date ($arriveBy=true)
//          STRING | format: 06-28-2024
// * time = departure time ($arriveBy=false) / arrival time ($arriveBy=true)
//          STRING | format: 9:06am // 7:06pm // 19:06
// * wheelchair = BOOL | whether the trip must be wheelchair accessible
// * fromPlace = STRING | latitude, longitude pair in degrees || stop id
// * toPlace = STRING | latitude, longitude pair in degrees || stop id
// * maxPreTransitTime = INTEGER | maximum time in seconds for firstLegModes
//   maxPostTransitTime = [..] for lastLegModes, defaults=$maxPreTransitTime
json::value graph::operator()(json::value const& query) const {
  return json::value{{"success", "true"}};
}

}  // namespace icc::ep