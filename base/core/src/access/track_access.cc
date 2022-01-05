#include "motis/core/access/track_access.h"

#include <algorithm>

namespace motis {

uint32_t get_track_string_idx(schedule const& sched, uint32_t const track_idx,
                              day_idx_t const day_idx) {
  if (track_idx == 0U) {
    return 0;
  }

  auto const tracks = sched.tracks_[track_idx];
  auto const it = std::find_if(
      begin(tracks), end(tracks),
      [&](cista::pair<uint32_t, bitfield_idx_or_ptr> const& entry) {
        return entry.second->test(day_idx);
      });

  return it == end(tracks) ? 0 : it->first;
}

mcd::string const& get_track_name(schedule const& sched,
                                  uint32_t const track_idx,
                                  day_idx_t const day_idx) {
  return *sched.string_mem_.at(get_track_string_idx(sched, track_idx, day_idx));
}

}  // namespace motis
