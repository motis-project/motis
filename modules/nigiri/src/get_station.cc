#include "motis/nigiri/trip_to_connection.h"

#include "motis/nigiri/nigiri_to_motis_journey.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/nigiri/resolve_run.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;
namespace mm = motis::module;

namespace motis::nigiri {

motis::module::msg_ptr get_station(tag_lookup const& tags,
                                   ::nigiri::timetable const& tt,
                                   ::nigiri::rt_timetable const* rtt,
                                   motis::module::msg_ptr const& msg) {
  using railviz::RailVizStationRequest;
  auto const req = motis_content(RailVizStationRequest, msg);
  CISTA_UNUSED_PARAM(tags)  // TODO(felix)
  CISTA_UNUSED_PARAM(tt)  // TODO(felix)
  CISTA_UNUSED_PARAM(rtt)  // TODO(felix)
  return mm::make_success_msg();
}

}  // namespace motis::nigiri