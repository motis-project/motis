#include "motis/nigiri/trip_to_connection.h"

#include "motis/nigiri/nigiri_to_motis_journey.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/core/journey/extern_trip.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;

namespace motis::nigiri {

n::rt::run resolve_run(tag_lookup const& tags, extern_trip const& et) {
  auto const [tag, trip_id] = split_tag_and_location_id(et.id_);
  auto const src = tags.get_src(tag);
  auto const day = date::sys_days{
      std::chrono::time_point_cast<date::days>(to_nigiri_unixtime(et.time_))};

  return {};
}

motis::module::msg_ptr trip_to_connection(tag_lookup const& tags,
                                          ::nigiri::timetable const& tt,
                                          ::nigiri::rt_timetable const* rtt,
                                          motis::module::msg_ptr const& msg) {}

}  // namespace motis::nigiri