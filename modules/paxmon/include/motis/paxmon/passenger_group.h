#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

struct edge;

struct data_source {
  CISTA_COMPARABLE()

  std::uint64_t primary_ref_{};
  std::uint64_t secondary_ref_{};
};

struct passenger_group {
  passenger_group_index id_{};
  data_source source_{};
  std::uint16_t passengers_{1};
};

inline passenger_group make_passenger_group(data_source const source,
                                            std::uint16_t const passengers,
                                            passenger_group_index id = 0) {
  return passenger_group{id, source, passengers};
}

}  // namespace motis::paxmon
