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

motis::module::msg_ptr trip_to_connection(tag_lookup const& tags,
                                          ::nigiri::timetable const& tt,
                                          ::nigiri::rt_timetable const* rtt,
                                          motis::module::msg_ptr const& msg) {
  auto const et = to_extern_trip(motis_content(TripId, msg));
  auto const r = resolve_run(tags, tt, et);
  if (!r.valid()) {
    LOG(logging::error) << "unable to find trip " << et.to_str();
    throw utl::fail("unable to resolve {}, departure {} at {}", et.id_,
                    et.time_, et.station_id_);
  }

  auto fr = n::rt::frun{tt, rtt, r};
  fr.stop_range_.to_ = fr.size();
  fr.stop_range_.from_ = 0U;
  auto const from_l = fr[0];
  auto const to_l = fr[fr.size() - 1U];
  auto const start_time = from_l.time(n::event_type::kDep);
  auto const dest_time = to_l.time(n::event_type::kArr);
  auto const j = nigiri_to_motis_journey(
      tt, rtt, tags,
      n::routing::journey{.legs_ = {n::routing::journey::leg{
                              n::direction::kForward, from_l.get_location_idx(),
                              to_l.get_location_idx(), start_time, dest_time,
                              n::routing::journey::run_enter_exit{
                                  fr,  // NOLINT(cppcoreguidelines-slicing)
                                  fr.first_valid(), fr.last_valid()}}},
                          .start_time_ = start_time,
                          .dest_time_ = dest_time,
                          .dest_ = to_l.get_location_idx(),
                          .transfers_ = 0U});

  mm::message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union());
  return make_msg(fbb);
}

}  // namespace motis::nigiri