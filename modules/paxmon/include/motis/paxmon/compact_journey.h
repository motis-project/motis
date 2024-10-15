#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "cista/hashing.h"
#include "cista/reflection/comparable.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

struct journey_leg {
  CISTA_COMPARABLE()

  cista::hash_t hash() const {
    return cista::build_hash(trip_idx_, enter_station_id_, exit_station_id_,
                             enter_time_, exit_time_);
  }

  trip_idx_t trip_idx_{0};
  unsigned enter_station_id_{0};
  unsigned exit_station_id_{0};
  motis::time enter_time_{0};
  motis::time exit_time_{0};
  std::optional<transfer_info> enter_transfer_;
};

struct compact_journey {
  CISTA_COMPARABLE()

  inline unsigned start_station_id() const {
    return legs_.front().enter_station_id_;
  }

  inline unsigned destination_station_id() const {
    return legs_.back().exit_station_id_;
  }

  inline duration scheduled_duration() const {
    return !legs_.empty() ? legs_.back().exit_time_ - legs_.front().enter_time_
                          : 0;
  }

  inline time scheduled_departure_time() const {
    return !legs_.empty() ? legs_.front().enter_time_ : INVALID_TIME;
  }

  inline time scheduled_arrival_time() const {
    return !legs_.empty() ? legs_.back().exit_time_ : INVALID_TIME;
  }

  cista::hash_t hash() const {
    auto h = cista::BASE_HASH;
    for (auto const& leg : legs_) {
      h = cista::hash_combine(h, leg.hash());
    }
    return h;
  }

  std::vector<journey_leg> legs_;
};

}  // namespace motis::paxmon
