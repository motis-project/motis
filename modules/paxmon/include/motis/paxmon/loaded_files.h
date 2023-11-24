#pragma once

#include <cstddef>
#include <filesystem>

#include "motis/core/common/unixtime.h"

#include "motis/paxmon/loader/capacities/load_capacities.h"

namespace motis::paxmon {

struct loaded_journey_file {
  std::filesystem::path path_;
  unixtime last_modified_{};

  std::size_t matched_journeys_{};
  std::size_t unmatched_journeys_{};
  std::size_t unmatched_journeys_rerouted_{};

  std::size_t matched_pax_{};
  std::size_t unmatched_pax_{};
  std::size_t unmatched_pax_rerouted_{};
};

struct loaded_capacity_file {
  std::filesystem::path path_;
  unixtime last_modified_{};

  loader::capacities::csv_format format_{};
  std::size_t loaded_entry_count_{};
  std::size_t skipped_entry_count_{};
  std::size_t station_not_found_count_{};
};

}  // namespace motis::paxmon
