#pragma once

#include <memory>
#include <string_view>

#include "motis/elevators/elevators.h"
#include "motis/fwd.h"

namespace motis {

std::unique_ptr<elevators> update_elevators(config const&,
                                            data const&,
                                            std::string_view fasta_json,
                                            nigiri::rt_timetable&);

}  // namespace motis