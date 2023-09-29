#pragma once

#include <filesystem>
#include <string_view>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace motis::transfers::restrictions {

// Requests availability times of the mobility service and saves them in
// csv-format (given as fs::path).
void load_mobility_service_availability(std::filesystem::path const&);

// Updates the timetable and adds to it the restriction of mobility service
// availabilities.
// The mobility service availability is applied to parent stations.
// There can be any number of availability periods per day.
//
// Requirement.
// 1st: data is in CSV format at the given path.
// 2nd: data file requires the following format (name, weekday, from, to)
void update_timetable_with_mobility_service_availability(
    ::nigiri::timetable&, std::filesystem::path const&);

}  // namespace motis::transfers::restrictions
