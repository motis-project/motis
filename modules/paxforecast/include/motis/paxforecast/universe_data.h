#pragma once

#include "motis/paxmon/metrics.h"

#include "motis/paxforecast/measures/measures.h"
#include "motis/paxforecast/statistics.h"

namespace motis::paxforecast {

struct universe_data {
  measures::measure_collection measures_;
  motis::paxmon::metrics<tick_statistics> metrics_;
};

}  // namespace motis::paxforecast
