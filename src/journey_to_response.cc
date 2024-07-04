#include "icc/journey_to_response.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "utl/enumerate.h"

namespace n = nigiri;

namespace icc {

std::int64_t to_unixtime_ms(n::unixtime_t const t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             t.time_since_epoch())
      .count();
}

std::int64_t to_seconds(n::i32_minutes const t) {
  return std::chrono::duration_cast<std::chrono::seconds>(t).count();
}

api::Itinerary journey_to_response(n::timetable const& tt,
                                   n::rt_timetable const* rtt,
                                   n::routing::journey const& j) {
  auto legs = std::vector<api::Leg>{};
  for (auto const [i, leg] : utl::enumerate(j.legs_)) {
    std::visit(
        utl::overloaded{[&](n::routing::journey::run_enter_exit const& t) {},
                        [&](n::footpath const fp) {},  //
                        [&](n::routing::offset const x) {}},
        leg.uses_);
    legs.emplace_back(api::Leg{
        .startTime_ = to_unixtime_ms(leg.dep_time_),
        .endTime_ = to_unixtime_ms(leg.arr_time_),
        .departureDelay_ = 0,  // TODO
        .arrivalDelay_ = 0,  // TODO
        .realTime_ = false,  // TODO
        .distance_ = 0.0,  // TODO
        .interlineWithPreviousLeg_ = false,  // TODO
        .route_ = "",  // TODO
        .headsign_ = "",  // TODO
        .agencyName_ = "",  // TODO
        .agencyUrl_ = "",  // TODO
        .routeColor_ = "",  // TODO
        .routeTextColor_ = "",  // TODO
        .routeType_ = "",  // TODO
        .routeId_ = "",  // TODO
        .agencyId_ = "",  // TODO
        .tripId_ = "",  // TODO
        .serviceDate_ = "",  // TODO
        .duration_ = to_seconds(leg.arr_time_ - leg.dep_time_),
        .intermediateStops_ = {},  // TODO
        .from_ = api::Place{},  // TODO
        .to_ = api::Place{},  // TODO
        .legGeometry_ = api::EncodedPolyline{},  // TODO
        .steps_ = {},  // TODO
        .mode_ = api::ModeEnum::TRANSIT  // TODO
    });
  }
  return {.duration_ = to_seconds(j.arrival_time() - j.departure_time()),
          .startTime_ = to_unixtime_ms(j.departure_time()),
          .endTime_ = to_unixtime_ms(j.arrival_time()),
          .walkTime_ = 0,  // TODO
          .walkDistance_ = 0,  // TODO
          .transfers_ = 0,  // TODO
          .legs_ = std::move(legs)};
}

}  // namespace icc