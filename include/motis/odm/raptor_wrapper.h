#pragma once

#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace motis::odm {

struct raptor_result {
  explicit raptor_result(
      nigiri::routing::routing_result<nigiri::routing::raptor_stats>&&);

  nigiri::pareto_set<nigiri::routing::journey> journeys_;
  nigiri::interval<nigiri::unixtime_t> interval_;
  nigiri::routing::search_stats search_stats_;
  nigiri::routing::raptor_stats algo_stats_;
};

raptor_result raptor_wrapper(
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    nigiri::routing::query,
    nigiri::direction,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace motis::odm