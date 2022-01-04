#include "motis/core/journey/journey_util.h"

#include <algorithm>
#include <numeric>

#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

namespace motis {

duration_t get_duration(journey const& journey) {
  if (!journey.stops_.empty()) {
    return (journey.stops_.back().arrival_.timestamp_ -
            journey.stops_.front().departure_.timestamp_) /
           60;
  }
  return 0;
}

uint16_t get_transfers(journey const& journey) {
  auto const exits =
      std::count_if(begin(journey.stops_), end(journey.stops_),
                    [](journey::stop const& s) { return s.exit_; });
  return exits == 0 ? 0 : exits - 1;
}

uint16_t get_accessibility(journey const& journey) {
  return std::accumulate(begin(journey.transports_), end(journey.transports_),
                         0, [](uint16_t acc, journey::transport const& t) {
                           return acc + t.mumo_accessibility_;
                         });
}

}  // namespace motis
