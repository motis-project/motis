#include "motis/core/access/transfer_time.h"

namespace motis {

std::optional<duration> get_transfer_time_between_platforms(
    station const& from_station, std::optional<uint16_t> const from_platform,
    station const& to_station, std::optional<uint16_t> const to_platform) {
  if (from_station.index_ == to_station.index_) {
    return from_station.get_transfer_time_between_platforms(from_platform,
                                                            to_platform);
  } else {
    for (auto const& fp : from_station.outgoing_footpaths_) {
      if (fp.to_station_ == to_station.index_) {
        return fp.duration_;
      }
    }
    return {};
  }
}

}  // namespace motis
