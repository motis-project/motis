#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity.h"

namespace motis::paxmon::loader::capacities {

enum class csv_format {
  TRIP,
  RIS_SERVICE_VEHICLES,
  FZG_KAP,
  FZG_GRUPPE,
  GATTUNG,
  BAUREIHE
};

enum class csv_separator { COMMA, SEMICOLON };

struct load_capacities_result {
  csv_format format_{};
  std::size_t loaded_entry_count_{};
  std::size_t skipped_entry_count_{};
  std::set<std::string> stations_not_found_;
};

load_capacities_result load_capacities(schedule const& sched,
                                       capacity_maps& caps,
                                       std::string_view const data);

load_capacities_result load_capacities_from_file(
    schedule const& sched, capacity_maps& caps,
    std::string const& capacity_file, std::string const& match_log_file = "");

}  // namespace motis::paxmon::loader::capacities
