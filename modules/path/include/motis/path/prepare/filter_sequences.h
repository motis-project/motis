#pragma once

#include <string>
#include <vector>

#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

void filter_sequences(std::vector<std::string> const& filters,
                      std::vector<station_seq>& sequences);

}  // namespace motis::path
