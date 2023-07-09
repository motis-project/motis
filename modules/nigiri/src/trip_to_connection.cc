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
#include "motis/nigiri/location.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;
namespace mm = motis::module;

namespace motis::nigiri {

n::rt::run resolve_run(tag_lookup const& tags, n::timetable const& tt,
                       extern_trip const& et) {
  auto const [tag, trip_id] = split_tag_and_location_id(et.id_);
  auto const src = tags.get_src(tag);
  auto const dep_time = to_nigiri_unixtime(et.time_);
  auto const day_idx = tt.day_idx(
      date::sys_days{std::chrono::time_point_cast<date::days>(dep_time)});

  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](n::pair<n::trip_id_idx_t, n::trip_idx_t> const& a,
          n::string const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(src, std::string_view{b});
      });

  auto const id_matches = [src, trip_id = trip_id,
                           &tt](n::trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t_idx, stop_range] :
         tt.trip_transport_ranges_[i->second]) {
      auto const t = n::transport{t_idx, day_idx};
      if (dep_time != tt.event_time(t, stop_range.from_, n::event_type::kDep)) {
        continue;
      }

      auto const& traffic_days =
          tt.bitfields_[tt.transport_traffic_days_[t_idx]];
      if (!traffic_days.test(to_idx(day_idx))) {
        continue;
      }

      return n::rt::run{.t_ = t, .stop_range_ = stop_range};
    }
  }

  return {};
}

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

  auto const fr = n::rt::frun{tt, rtt, r};
  auto const from_l = fr[0];
  auto const to_l = fr[fr.size() - 1U];
  auto const start_time = from_l.time(n::event_type::kDep);
  auto const dest_time = to_l.time(n::event_type::kArr);
  auto const j = nigiri_to_motis_journey(
      tt, rtt, tags,
      n::routing::journey{
          .legs_ = {n::routing::journey::leg{
              n::direction::kForward, from_l.get_location_idx(),
              to_l.get_location_idx(), start_time, dest_time,
              n::routing::journey::run_enter_exit{
                  fr, r.stop_range_.from_,
                  static_cast<n::stop_idx_t>(r.stop_range_.to_ - 1U)}}},
          .start_time_ = start_time,
          .dest_time_ = dest_time,
          .transfers_ = 0U});

  mm::message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union());
  return make_msg(fbb);
}

}  // namespace motis::nigiri