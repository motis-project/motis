#include "./population.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "fmt/core.h"

#include "nigiri/timetable.h"

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/read_file.h"
#include "utl/verify.h"

#include "motis/types.h"

namespace n = nigiri;
namespace po = boost::program_options;

namespace motis {

namespace {

pop_weight parse_pop_weight(std::string_view const s,
                            std::string_view const opt) {
  utl::verify(s == "high" || s == "low",
              R"(--{} must be "high" or "low", got "{}")", opt, s);
  return s == "high" ? pop_weight::kHigh : pop_weight::kLow;
}

geo::grid<std::uint64_t> load_population_grid(std::string const& s) {
  auto const file_content = utl::read_file(s.c_str());
  utl::verify(file_content.has_value(),
              "could not read population grid file at {}", s.c_str());
  auto grid = geo::parse_eurostat_population_grid(*file_content);
  utl::erase_if(grid, [](auto const c) { return c.data_ == 0UL; });
  utl::verify(!grid.empty(),
              "population grid file at {} contains no populated cells",
              s.c_str());
  return grid;
}

}  // namespace

std::string_view description(pop_weight const w) {
  return w == pop_weight::kHigh
             ? "weighted towards high population (any cell can be drawn, "
               "likelihood proportional to its population)"
             : "weighted towards low population (any cell can be drawn, "
               "the k-th least populated cell is as likely as the k-th most "
               "populated one is for high population)";
}

void population_sampler::add_options(po::options_description& desc) {
  desc.add_options()  //
      ("population_grid",
       po::value<std::string>()->notifier(
           [this](std::string const& s) { grid_ = load_population_grid(s); }),
       "path to a CSV file containing a EUROSTAT population grid; requires "
       "--population_from and/or --population_to to say which end of a query "
       "is drawn from it")  //
      ("population_from",
       po::value<std::string>()->notifier([this](std::string_view const s) {
         from_ = parse_pop_weight(s, "population_from");
       }),
       "weight the origin towards \"high\" or \"low\" population: every "
       "populated grid cell with an eligible stop can be drawn, only its "
       "likelihood changes - proportional to the cell's population for "
       "\"high\"; the mirror image for \"low\": the k-th least populated cell "
       "is as likely as the k-th most populated cell is for \"high\"; "
       "requires --population_grid, not supported with FLEX")  //
      ("population_to",
       po::value<std::string>()->notifier([this](std::string_view const s) {
         to_ = parse_pop_weight(s, "population_to");
       }),
       "weight the destination the same way, overriding the default "
       "lower-bounds rank without needing --lb_rank 0; requires "
       "--population_grid and conflicts with an explicit --lb_rank 1/"
       "--geo_rank, which derive `to` from `from`. Combined with "
       "--population_from this spans the four pairings: high->low is a funnel, "
       "low->high its reverse, high->high and low->low weight both ends the "
       "same way");
}

void population_sampler::verify(bool const use_flex,
                                bool const use_geo_rank,
                                bool const explicit_lb_rank) const {
  // the notifier rejects grids without populated cells
  auto const has_population_grid = !grid_.empty();
  utl::verify((!from_ && !to_) || has_population_grid,
              "--population_from/--population_to require --population_grid");
  utl::verify(!has_population_grid || from_ || to_,
              "--population_grid requires --population_from and/or "
              "--population_to to say which end is drawn by population");
  utl::verify(!from_ || !use_flex,
              "--population_from cannot be combined with --modes FLEX: FLEX "
              "draws `from` from the flex areas");
  utl::verify(!to_ || !use_geo_rank,
              "--population_to cannot be combined with --geo_rank: both "
              "decide how `to` is picked");
  utl::verify(!to_ || !explicit_lb_rank,
              "--population_to cannot be combined with --lb_rank 1: both "
              "decide how `to` is picked. --population_to already overrides "
              "the default rank, so just drop --lb_rank");
}

void population_sampler::match_stops(
    n::timetable const& tt, std::vector<n::location_idx_t> const& stops) {
  if (grid_.empty()) {
    return;
  }

  auto const resolution = grid_.front().id_.resolution_;
  utl::verify(utl::all_of(grid_,
                          [&](auto const& gc) {
                            return gc.id_.resolution_ == resolution;
                          }),
              "population grid cells must all have the same resolution");

  auto stops_by_cell =
      hash_map<geo::inspire_cell, std::vector<n::location_idx_t>>{};
  for (auto const l : stops) {
    if (!tt.location_routes_[l].empty()) {
      stops_by_cell[geo::inspire_cell_of(tt.locations_.coordinates_[l],
                                         resolution)]
          .emplace_back(l);
    }
  }

  auto const n_before = grid_.size();
  auto cells_with_stops = geo::grid<std::uint64_t>{};
  for (auto const& gc : grid_) {
    if (auto const it = stops_by_cell.find(gc.id_); it != end(stops_by_cell)) {
      cells_with_stops.emplace_back(gc);
      cell_stops_.emplace_back(std::move(it->second));
      stops_by_cell.erase(it);  // a cell listed twice keeps its stops once
    }
  }
  grid_ = std::move(cells_with_stops);
  auto const n_discarded = n_before - grid_.size();

  utl::verify(!grid_.empty(),
              "can not generate queries: no population grid cells with "
              "eligible stops remain");

  auto const total_population = std::accumulate(
      grid_.begin(), grid_.end(), std::uint64_t{0U},
      [](auto const acc, auto const& gc) { return acc + gc.data_; });
  fmt::println(
      "population grid: discarded {}/{} cells with no eligible stops, {} "
      "cells with total population {} remain for query generation",
      n_discarded, n_before, grid_.size(), total_population);

  auto const pop = [&](std::size_t const i) { return grid_[i].data_; };
  auto const cumulative_weights = [&](auto&& weight) {
    auto v = std::vector<double>(grid_.size());
    auto sum = 0.0;
    for (auto i = 0UL; i != grid_.size(); ++i) {
      sum += weight(i);
      v[i] = sum;
    }
    return v;
  };
  if (from_ == pop_weight::kHigh || to_ == pop_weight::kHigh) {
    cumulative_weight_high_ = cumulative_weights(
        [&](std::size_t const i) { return static_cast<double>(pop(i)); });
  }
  if (from_ == pop_weight::kLow || to_ == pop_weight::kLow) {
    // rank mirror of kHigh: the k-th least populated cell weighs as much as
    // the k-th most populated one, i.e. kHigh's weights in reverse order;
    // cells with equal population share the mean of their mirrored weights,
    // so the order among them does not matter
    auto by_pop = std::vector<std::size_t>(grid_.size());
    std::iota(begin(by_pop), end(by_pop), std::size_t{0U});
    utl::sort(by_pop,
              [&](auto const a, auto const b) { return pop(a) < pop(b); });
    auto const m = by_pop.size();
    auto mirrored = std::vector<double>(m);
    for (auto first = 0UL; first != m;) {
      auto last = first;
      auto sum = 0.0;
      for (; last != m && pop(by_pop[last]) == pop(by_pop[first]); ++last) {
        sum += static_cast<double>(pop(by_pop[m - 1U - last]));
      }
      for (auto k = first; k != last; ++k) {
        mirrored[by_pop[k]] = sum / static_cast<double>(last - first);
      }
      first = last;
    }
    cumulative_weight_low_ =
        cumulative_weights([&](std::size_t const i) { return mirrored[i]; });
  }
}

std::vector<n::location_idx_t> const& population_sampler::random_cell(
    pop_weight const w, double const u) const {
  auto const& cumulative =
      w == pop_weight::kHigh ? cumulative_weight_high_ : cumulative_weight_low_;
  auto const r = u * cumulative.back();
  auto const cell = std::upper_bound(cumulative.begin(), cumulative.end(), r);
  auto const cell_idx = std::min(
      static_cast<std::size_t>(std::distance(cumulative.begin(), cell)),
      cumulative.size() - 1U);  // guard against rounding
  return cell_stops_[cell_idx];
}

}  // namespace motis
