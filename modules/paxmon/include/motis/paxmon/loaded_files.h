#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "motis/core/common/unixtime.h"

#include "motis/paxmon/loader/capacities/load_capacities.h"
#include "motis/paxmon/loader/unmatched_journey.h"

namespace motis::paxmon {

struct loaded_journey_file {
  std::filesystem::path path_;
  unixtime last_modified_{};

  std::size_t matched_journey_count_{};
  std::size_t unmatched_journey_count_{};
  std::size_t unmatched_journey_rerouted_count_{};

  std::size_t matched_group_count_{};
  std::size_t unmatched_group_count_{};
  std::size_t unmatched_group_rerouted_count_{};

  std::size_t matched_pax_count_{};
  std::size_t unmatched_pax_count_{};
  std::size_t unmatched_pax_rerouted_count_{};

  std::vector<loader::unmatched_journey> unmatched_journeys_;
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
