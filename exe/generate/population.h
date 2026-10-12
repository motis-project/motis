#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "boost/program_options.hpp"

#include "geo/grid.h"

#include "nigiri/types.h"

#include "motis/fwd.h"

namespace motis {

// which end of the population distribution a stop is drawn from
enum class pop_weight { kHigh, kLow };

// how a stop drawn with the given weight is picked, for the generate log
std::string_view description(pop_weight);

// draws stops weighted by the population of the EUROSTAT grid cell they lie in
struct population_sampler {
  // adds --population_grid, --population_from and --population_to
  void add_options(boost::program_options::options_description&);

  // rejects option combinations that contradict each other: explicit_lb_rank
  // is set if --lb_rank 1 was given explicitly (not just by default)
  void verify(bool use_flex, bool use_geo_rank, bool explicit_lb_rank) const;

  // keeps only the grid cells that contain at least one of the given stops
  // with a route and computes their selection weights, no-op without a grid
  void match_stops(nigiri::timetable const&,
                   std::vector<nigiri::location_idx_t> const& stops);

  // randomizes a grid cell by its selection weight (u in [0, 1) scaled to
  // the total weight followed by a binary search over the running totals)
  // and returns the cell's eligible stops
  std::vector<nigiri::location_idx_t> const& random_cell(pop_weight,
                                                         double u) const;

  std::optional<pop_weight> from_;
  std::optional<pop_weight> to_;

  geo::grid<std::uint64_t> grid_;

  // for each population grid cell that contains at least one eligible stop
  // (i.e. a stop within the geo bounds and the ODM bounds (if given) with at
  // least one route): the eligible stops within the cell, used to weight
  // random stop selection by population
  std::vector<std::vector<nigiri::location_idx_t>> cell_stops_;

  // running totals of the per-cell selection weights, indexed like grid_ and
  // cell_stops_: population for kHigh, its rank mirror for kLow (only computed
  // for the weightings in use)
  std::vector<double> cumulative_weight_high_;
  std::vector<double> cumulative_weight_low_;
};

}  // namespace motis
