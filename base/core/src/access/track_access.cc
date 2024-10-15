#include "motis/core/access/track_access.h"

#include <algorithm>
#include <iterator>

namespace motis {

std::optional<uint16_t> get_track_index(schedule const& sched,
                                        std::string_view const track_name) {
  auto const it =
      std::find_if(begin(sched.tracks_), end(sched.tracks_),
                   [&track_name](auto const& t) { return t == track_name; });
  if (it != end(sched.tracks_)) {
    return static_cast<uint16_t>(std::distance(begin(sched.tracks_), it));
  } else {
    return {};
  }
}

}  // namespace motis
