#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"

#include "icc-api/icc-api.h"

namespace icc {

api::Itinerary journey_to_response(nigiri::routing::journey const&);

}  // namespace icc