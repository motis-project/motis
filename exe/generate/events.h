#pragma once

#include <vector>

#include "boost/program_options.hpp"

#include "date/date.h"

#include "nigiri/types.h"

#include "motis/fwd.h"

namespace motis {

// draws origins weighted by their number of departures and destinations
// weighted by their number of arrivals within [first_day, last_day)
struct event_sampler {
  // adds --event_weighted
  void add_options(boost::program_options::options_description&);

  // rejects the combination with --modes FLEX, which draws `from` from the
  // flex areas, and with --population_from/--population_to: both decide how
  // `from` and `to` are picked
  void verify(bool use_flex, bool use_population) const;

  // counts departure and arrival events per stop within
  // [first_day, last_day), no-op without --event_weighted
  void count_events(nigiri::timetable const&,
                    std::vector<nigiri::location_idx_t> const& stops,
                    date::sys_days first_day,
                    date::sys_days last_day);

  // picks one of the stops given to count_events with probability
  // proportional to its departure weight, falling back to a uniform pick from
  // `stops` if there are no departures at all
  nigiri::location_idx_t random_from(
      nigiri::timetable const&,
      std::vector<nigiri::location_idx_t> const& stops) const;

  // picks one of `stops` with probability proportional to its arrival weight,
  // falling back to a uniform pick from `stops` if there are no arrivals at
  // all
  nigiri::location_idx_t random_to(
      nigiri::timetable const&,
      std::vector<nigiri::location_idx_t> const& stops) const;

  bool enabled_{false};

  // number of departure/arrival events per location within
  // [first_day, last_day), accounting for each transport's traffic day
  // bitfield
  std::vector<double> dep_weight_;
  std::vector<double> arr_weight_;

  // the stops given to count_events and the cumulative departure weight over
  // them, used to sample from_place in O(log n) via binary search
  std::vector<nigiri::location_idx_t> stops_;
  std::vector<double> dep_cum_;
};

}  // namespace motis
