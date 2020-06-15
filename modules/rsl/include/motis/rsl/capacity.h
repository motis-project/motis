#pragma once

#include <cstdint>

#include "utl/struct/comparable.h"

#include "motis/data.h"
#include "motis/hash_map.h"

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/schedule.h"

namespace motis::rsl {

struct train_name {
  MAKE_COMPARABLE()

  cista::hash_t hash() const { return cista::build_hash(category_, train_nr_); }

  mcd::string category_;
  std::uint32_t train_nr_;
};

using capacity_map_t = mcd::hash_map<train_name, std::uint16_t>;

capacity_map_t load_capacities(std::string const& capacity_file);

std::uint16_t guess_capacity(schedule const& sched, light_connection const& lc);

std::uint16_t get_capacity(schedule const& sched, capacity_map_t const& map,
                           light_connection const& lc);

}  // namespace motis::rsl
