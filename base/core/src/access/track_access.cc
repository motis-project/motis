#include "motis/core/access/track_access.h"

#include <algorithm>

namespace motis {

uint32_t get_track_string_idx(schedule const& sched,
                              generic_light_connection const& glcon,
                              event_type const ev_type) {
  auto const track_idx = glcon.full_con().get_track(ev_type);
  if (track_idx == 0U) {
    return 0;
  }

  auto const day_idx = glcon.event_time(ev_type).day();
  auto const tracks = sched.tracks_[track_idx];
  auto const it = std::find_if(
      begin(tracks), end(tracks),
      [&](cista::pair<uint32_t, bitfield_idx_or_ptr> const& entry) {
        return entry.second->test(day_idx);
      });

  return it == end(tracks) ? 0 : it->first;
}

mcd::string const& get_track_name(schedule const& sched,
                                  generic_light_connection const& glcon,
                                  event_type const ev_type) {
  return *sched.string_mem_.at(get_track_string_idx(sched, glcon, ev_type));
}

}  // namespace motis
