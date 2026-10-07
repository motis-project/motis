#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>

#include "conf/configuration.h"

#include "boost/url/url.hpp"

#include "geo/grid.h"

#include "tg.h"

#include "nigiri/common/interval.h"
#include "nigiri/flex.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/raii.h"
#include "utl/read_file.h"
#include "utl/verify.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/constants.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/odm/bounds.h"
#include "motis/point_rtree.h"
#include "motis/tag_lookup.h"

#include "./flags.h"

namespace n = nigiri;
namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace motis {

constexpr auto kMinRank = 16U;
constexpr auto kMaxPlaceAttempts = 1000U;

constexpr auto kEuropeBounds = R"({
  "type": "Polygon",
  "coordinates":
    [[ [ -11, 72 ], [ -11, 36 ], [ 32, 36 ], [ 32, 72 ], [ -11, 72 ] ]]
})";

static std::atomic_uint32_t seed{0U};

std::uint32_t rand_in(std::uint32_t const from, std::uint32_t const to) {
  auto a = ++seed;
  a = (a ^ 61U) ^ (a >> 16U);
  a = a + (a << 3U);
  a = a ^ (a >> 4U);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15U);
  return from + (a % (to - from));
}

std::uint64_t rand_in(std::uint64_t const from, std::uint64_t const to) {
  auto const hi = rand_in(0U, std::numeric_limits<std::uint32_t>::max());
  auto const lo = rand_in(0U, std::numeric_limits<std::uint32_t>::max());
  auto const combined =
      (static_cast<std::uint64_t>(hi) << 32U) | static_cast<std::uint64_t>(lo);
  return from + (combined % (to - from));
}

// uniformly distributed in [0, 1): 53 random bits fill a double's mantissa
double rand_unit() {
  return static_cast<double>(
             rand_in(std::uint64_t{0U}, std::uint64_t{1U} << 53U)) *
         0x1.0p-53;
}

template <typename It>
It rand_in(It const begin, It const end) {
  return std::next(
      begin,
      rand_in(0U, static_cast<std::uint32_t>(std::distance(begin, end))));
}

template <typename Collection>
Collection::value_type rand_in(Collection const& c) {
  using std::begin;
  using std::end;
  utl::verify(!c.empty(), "empty collection");
  return *rand_in(begin(c), end(c));
}

n::location_idx_t random_stop(n::timetable const& tt,
                              std::vector<n::location_idx_t> const& stops) {
  auto s = n::location_idx_t::invalid();
  do {
    s = rand_in(stops);
  } while (tt.location_routes_[s].empty());
  return s;
}

// which end of the population distribution a stop is drawn from
enum class pop_weight { kHigh, kLow };

int generate(int ac, char** av) {
  auto data_path = fs::path{"data"};
  auto n = 100U;
  auto first_day = std::optional<date::sys_days>{};
  auto last_day = std::optional<date::sys_days>{};
  auto time_of_day = std::optional<std::uint32_t>{};
  auto modes = std::optional<std::vector<api::ModeEnum>>{};
  auto max_dist = 800.0;  // m
  auto use_walk = false;
  auto use_bike = false;
  auto use_car = false;
  auto use_odm = false;
  auto use_flex = false;
  auto lb_rank = true;
  auto geo_rank = std::optional<std::uint64_t>{};
  auto population_from = std::optional<pop_weight>{};
  auto population_to = std::optional<pop_weight>{};
  tg_geom* bounds{nullptr};
  auto const free_bounds = utl::make_finally([&]() { tg_geom_free(bounds); });
  auto population_grid = geo::grid<std::uint64_t>{};
  auto master_params = api::plan_params{};

  auto const parse_date = [](std::string_view const s) {
    std::stringstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in << s;
    auto d = date::sys_days{};
    in >> date::parse("%Y-%m-%d", d);
    return d;
  };

  auto const parse_first_day = [&](std::string_view const s) {
    first_day = parse_date(s);
  };

  auto const parse_last_day = [&](std::string_view const s) {
    last_day = parse_date(s);
  };

  auto const parse_modes = [&](std::string_view const s) {
    modes = std::vector<api::ModeEnum>{};
    if (s.contains("WALK")) {
      modes->emplace_back(api::ModeEnum::WALK);
      use_walk = true;
    }
    if (s.contains("BIKE")) {
      modes->emplace_back(api::ModeEnum::BIKE);
      use_bike = true;
    }
    if (s.contains("CAR")) {
      modes->emplace_back(api::ModeEnum::CAR);
      use_car = true;
    }
    if (s.contains("ODM")) {
      modes->emplace_back(api::ModeEnum::ODM);
      use_odm = true;
    }
    if (s.contains("RIDE_SHARING")) {
      modes->emplace_back(api::ModeEnum::RIDE_SHARING);
      use_odm = true;
    }
    if (s.contains("FLEX")) {
      modes->emplace_back(api::ModeEnum::FLEX);
      use_flex = true;
    }
  };

  auto const parse_time_of_day = [&](std::uint32_t const h) {
    time_of_day = h % 24U;
  };

  auto const parse_bounds = [&](std::string_view const s) {
    bounds = s == "europe" ? tg_parse_geojson(kEuropeBounds)
                           : tg_parse_geojsonn(s.data(), s.size());
    if (char const* err = tg_geom_error(bounds)) {
      throw utl::fail("unable to parse bounds GeoJSON: {}", err);
    }
  };

  auto const parse_pop_weight = [](std::string_view const s,
                                   std::string_view const opt) {
    utl::verify(s == "high" || s == "low",
                R"(--{} must be "high" or "low", got "{}")", opt, s);
    return s == "high" ? pop_weight::kHigh : pop_weight::kLow;
  };
  auto const parse_pop_from = [&](std::string_view const s) {
    population_from = parse_pop_weight(s, "population_from");
  };
  auto const parse_pop_to = [&](std::string_view const s) {
    population_to = parse_pop_weight(s, "population_to");
  };

  auto const parse_population_grid = [&](std::string const& s) {
    auto const file_content = utl::read_file(s.c_str());
    utl::verify(file_content.has_value(),
                "could not read population grid file at {}", s.c_str());
    population_grid = geo::parse_eurostat_population_grid(*file_content);
    utl::erase_if(population_grid, [](auto const c) { return c.data_ == 0UL; });
    utl::verify(!population_grid.empty(),
                "population grid file at {} contains no populated cells",
                s.c_str());
  };

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("n,n", po::value(&n)->default_value(n), "number of queries")  //
      ("first_day", po::value<std::string>()->notifier(parse_first_day),
       "first day of query generation, format: YYYY-MM-DD")  //
      ("last_day", po::value<std::string>()->notifier(parse_last_day),
       "last day of query generation, format: YYYY-MM-DD")  //
      ("time_of_day", po::value<std::uint32_t>()->notifier(parse_time_of_day),
       "fixes the time of day of all queries to the given number of hours "
       "after midnight, i.e., 0 - 23")  //
      ("modes,m", po::value<std::string>()->notifier(parse_modes),
       "comma-separated list of modes for first/last mile and "
       "direct (requires "
       "street routing), supported: WALK, BIKE, CAR, ODM")  //
      ("all,a",
       "requires OSM nodes to be accessible by all specified modes, otherwise "
       "OSM nodes accessible by at least one mode are eligible, only used for "
       "intermodal queries")  //
      ("max_dist", po::value(&max_dist)->default_value(max_dist),
       "maximum distance from a public transit stop in meters, only used for "
       "intermodal queries")  //
      ("max_travel_time",
       po::value<std::int64_t>()->notifier(
           [&](auto const v) { master_params.maxTravelTime_ = v; }),
       "sets maximum travel time of the queries")  //
      ("max_matching_distance",
       po::value(&master_params.maxMatchingDistance_)
           ->default_value(master_params.maxMatchingDistance_),
       "sets the maximum matching distance of the queries")  //
      ("fastest_direct_factor",
       po::value(&master_params.fastestDirectFactor_)
           ->default_value(master_params.fastestDirectFactor_),
       "sets fastest direct factor of the queries")  //
      ("lb_rank", po::value(&lb_rank)->default_value(lb_rank),
       "emit queries uniformly distributed over the lower bounds (lb) ranks, "
       "lb rank n:  2^n-th stop when sorting all stops by their lb value from "
       "the start (min. rank: 4, max. rank: derived from number of eligible "
       "stops)")  //
      ("geo_rank",
       po::value<std::uint64_t>()->notifier(
           [&](std::uint64_t const r) { geo_rank = r; }),
       "emit queries with geo-rank r, i.e., the target is the 2^r-th stop from "
       "the source in terms of geographical distance, overrides lb_rank")  //
      ("bounds,b", po::value<std::string>()->notifier(parse_bounds),
       "randomize locations within bounds, format: GeoJSON"
       "(shorthand for Europe \"-b europe\")")  //
      ("population_grid",
       po::value<std::string>()->notifier(parse_population_grid),
       "path to a CSV file containing a EUROSTAT population grid; requires "
       "--population_from and/or --population_to to say which end of a query "
       "is drawn from it")  //
      ("population_from", po::value<std::string>()->notifier(parse_pop_from),
       "weight the origin towards \"high\" or \"low\" population: every "
       "populated grid cell with an eligible stop can be drawn, only its "
       "likelihood changes - proportional to the cell's population for "
       "\"high\"; the mirror image for \"low\": the k-th least populated cell "
       "is as likely as the k-th most populated cell is for \"high\"; "
       "requires --population_grid, not supported with FLEX")  //
      ("population_to", po::value<std::string>()->notifier(parse_pop_to),
       "weight the destination the same way, overriding the default "
       "lower-bounds rank without needing --lb_rank 0; requires "
       "--population_grid and conflicts with an explicit --lb_rank 1/"
       "--geo_rank, which derive `to` from `from`. Combined with "
       "--population_from this spans the four pairings: high->low is a funnel, "
       "low->high its reverse, high->high and low->low weight both ends the "
       "same way");
  add_data_path_opt(desc, data_path);
  auto vm = parse_opt(ac, av, desc);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const has_population_grid = vm.count("population_grid") != 0U;
  utl::verify((!population_from && !population_to) || has_population_grid,
              "--population_from/--population_to require --population_grid");
  utl::verify(!has_population_grid || population_from || population_to,
              "--population_grid requires --population_from and/or "
              "--population_to to say which end is drawn by population");
  utl::verify(!population_from || !use_flex,
              "--population_from cannot be combined with --modes FLEX: FLEX "
              "draws `from` from the flex areas");
  utl::verify(!population_to || !geo_rank,
              "--population_to cannot be combined with --geo_rank: both "
              "decide how `to` is picked");
  utl::verify(!population_to || !(vm.count("lb_rank") != 0U &&
                                  !vm["lb_rank"].defaulted() && lb_rank),
              "--population_to cannot be combined with --lb_rank 1: both "
              "decide how `to` is picked. --population_to already overrides "
              "the default rank, so just drop --lb_rank");

  auto const c = config::read(data_path / "config.yml");
  utl::verify(c.timetable_.has_value(), "timetable required");
  utl::verify(!modes || c.use_street_routing(),
              "intermodal requires street routing");

  auto d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");

  fmt::println("Timetable ---\nn_locations: {}\nn_routes: {}\nn_trips: {}\n---",
               d.tt_->n_locations(), d.tt_->n_routes(), d.tt_->n_trips());

  first_day = first_day
                  ? d.tt_->date_range_.clamp(*first_day)
                  : std::chrono::time_point_cast<date::sys_days::duration>(
                        d.tt_->external_interval().from_);
  last_day = last_day ? d.tt_->date_range_.clamp(
                            std::max(*first_day + date::days{1U}, *last_day))
                      : d.tt_->date_range_.clamp(*first_day + date::days{14U});
  if (*first_day == *last_day) {
    fmt::println(
        "can not generate queries: date range [{}, {}] has zero length after "
        "clamping",
        *first_day, *last_day);
    return 1;
  }
  fmt::println("date range: [{}, {}], tt={}", *first_day, *last_day,
               d.tt_->external_interval());

  auto const in_bounds = [&](auto const& pos) {
    if (bounds == nullptr) {
      return true;
    }

    auto const point = tg_geom_new_point(tg_point{pos.lng(), pos.lat()});
    auto const result = tg_geom_within(point, bounds);
    tg_geom_free(point);
    return result;
  };

  auto const use_odm_bounds = modes && use_odm && d.odm_bounds_ != nullptr;
  auto node_rtree = point_rtree<osr::node_idx_t>{};
  if (modes) {
    if (modes->empty()) {
      fmt::println(
          "can not generate queries: provided modes option without valid "
          "mode");
      return 1;
    }
    std::cout << "modes:";
    for (auto const m : *modes) {
      std::cout << " " << m;
    }
    std::cout << "\n";

    master_params.directModes_ = *modes;
    master_params.preTransitModes_ = *modes;
    master_params.postTransitModes_ = *modes;

    auto const mode_match = [&](auto const node) {
      auto const can_walk = [&](auto const x) {
        return utl::any_of(d.w_->r_->node_ways_[x], [&](auto const w) {
          return d.w_->r_->way_properties_[w].is_foot_accessible();
        });
      };

      auto const can_bike = [&](auto const x) {
        return utl::any_of(d.w_->r_->node_ways_[x], [&](auto const w) {
          return d.w_->r_->way_properties_[w].is_bike_accessible();
        });
      };

      auto const can_car = [&](auto const x) {
        return utl::any_of(d.w_->r_->node_ways_[x], [&](auto const w) {
          return d.w_->r_->way_properties_[w].is_car_accessible();
        });
      };

      return vm.count("all") ? ((!use_walk || can_walk(node)) &&
                                (!use_bike || can_bike(node)) &&
                                (!(use_car || use_odm) || can_car(node)))
                             : ((use_walk && can_walk(node)) ||
                                (use_bike && can_bike(node)) ||
                                ((use_car || use_odm) && can_car(node)));
    };

    auto const in_odm_bounds = [&](auto const& pos) {
      return !use_odm_bounds || d.odm_bounds_->contains(pos);
    };

    for (auto i = osr::node_idx_t{0U}; i < d.w_->n_nodes(); ++i) {
      if (mode_match(i) && in_bounds(d.w_->get_node_pos(i)) &&
          in_odm_bounds(d.w_->get_node_pos(i))) {
        node_rtree.add(d.w_->get_node_pos(i), i);
      }
    }
  } else {
    fmt::println("station-to-station");
  }

  auto const master_stops = [&] {
    auto v = std::vector<n::location_idx_t>{};
    for (auto i = 0U; i != d.tt_->n_locations(); ++i) {
      auto const l = n::location_idx_t{i};

      if (!in_bounds(d.tt_->locations_.coordinates_[l]) ||
          (use_odm_bounds &&
           !d.odm_bounds_->contains(d.tt_->locations_.coordinates_[l]))) {
        continue;
      }
      v.emplace_back(l);
    }
    return v;
  }();

  if (bounds != nullptr) {
    fmt::println("in bounds: {}/{} stops", master_stops.size(),
                 d.tt_->n_locations());
  }

  // for each population grid cell that contains at least one eligible stop
  // (i.e. a stop within the geo bounds and the ODM bounds (if given) with at
  // least one route): the eligible stops within the cell, used to weight random
  // stop selection by population
  auto cell_stops = std::vector<std::vector<n::location_idx_t>>{};
  // running totals of the per-cell selection weights, indexed like
  // population_grid and cell_stops: population for kHigh, its rank mirror for
  // kLow (only computed for the weightings in use)
  auto cumulative_weight_high = std::vector<double>{};
  auto cumulative_weight_low = std::vector<double>{};
  if (!population_grid.empty()) {
    auto const resolution = population_grid.front().id_.resolution_;
    utl::verify(utl::all_of(population_grid,
                            [&](auto const& gc) {
                              return gc.id_.resolution_ == resolution;
                            }),
                "population grid cells must all have the same resolution");

    auto stops_by_cell =
        hash_map<geo::inspire_cell, std::vector<n::location_idx_t>>{};
    for (auto const l : master_stops) {
      if (!d.tt_->location_routes_[l].empty()) {
        stops_by_cell[geo::inspire_cell_of(d.tt_->locations_.coordinates_[l],
                                           resolution)]
            .emplace_back(l);
      }
    }

    auto const n_before = population_grid.size();
    auto cells_with_stops = geo::grid<std::uint64_t>{};
    for (auto const& gc : population_grid) {
      if (auto const it = stops_by_cell.find(gc.id_);
          it != end(stops_by_cell)) {
        cells_with_stops.emplace_back(gc);
        cell_stops.emplace_back(std::move(it->second));
        stops_by_cell.erase(it);  // a cell listed twice keeps its stops once
      }
    }
    population_grid = std::move(cells_with_stops);
    auto const n_discarded = n_before - population_grid.size();

    utl::verify(!population_grid.empty(),
                "can not generate queries: no population grid cells with "
                "eligible stops remain");

    auto const total_population = std::accumulate(
        population_grid.begin(), population_grid.end(), std::uint64_t{0U},
        [](auto const acc, auto const& gc) { return acc + gc.data_; });
    fmt::println(
        "population grid: discarded {}/{} cells with no eligible stops, {} "
        "cells with total population {} remain for query generation",
        n_discarded, n_before, population_grid.size(), total_population);

    auto const pop = [&](std::size_t const i) {
      return population_grid[i].data_;
    };
    auto const cumulative_weights = [&](auto&& weight) {
      auto v = std::vector<double>(population_grid.size());
      auto sum = 0.0;
      for (auto i = 0UL; i != population_grid.size(); ++i) {
        sum += weight(i);
        v[i] = sum;
      }
      return v;
    };
    if (population_from == pop_weight::kHigh ||
        population_to == pop_weight::kHigh) {
      cumulative_weight_high = cumulative_weights(
          [&](std::size_t const i) { return static_cast<double>(pop(i)); });
    }
    if (population_from == pop_weight::kLow ||
        population_to == pop_weight::kLow) {
      // rank mirror of kHigh: the k-th least populated cell weighs as much as
      // the k-th most populated one, i.e. kHigh's weights in reverse order;
      // cells with equal population share the mean of their mirrored weights,
      // so the order among them does not matter
      auto by_pop = std::vector<std::size_t>(population_grid.size());
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
      cumulative_weight_low =
          cumulative_weights([&](std::size_t const i) { return mirrored[i]; });
    }
  }

  struct flex_seed {
    geo::latlng from_;
    n::location_idx_t rank_stop_;
  };
  auto const flex_seeds = [&] {
    auto v = std::vector<flex_seed>{};
    if (use_flex) {
      for (auto i = 0U; i != d.tt_->flex_area_locations_.size(); ++i) {
        auto const a = n::flex_area_idx_t{i};
        auto const area_stops = d.tt_->flex_area_locations_[a];
        for (auto const l : area_stops) {
          v.emplace_back(d.tt_->locations_.coordinates_[l], l);
        }
        auto const& bbox = d.tt_->flex_area_bbox_[a];
        v.emplace_back(  // use flex area center
            geo::latlng{(bbox.min_.lat_ + bbox.max_.lat_) / 2.0,
                        (bbox.min_.lng_ + bbox.max_.lng_) / 2.0},
            area_stops.empty() ? n::location_idx_t::invalid() : area_stops[0]);
      }
      utl::verify(!v.empty(), "no flex areas in timetable");
      fmt::println("flex: {} areas, {} seeds (stops + area centers)",
                   d.tt_->flex_area_locations_.size(), v.size());
    }
    return v;
  }();

  // randomizes a grid cell by its selection weight (random number in
  // [0, total weight) followed by a binary search over the running totals),
  // then picks uniformly at random among the cell's eligible stops
  auto const random_population_weighted_stop = [&](pop_weight const w) {
    auto const& cumulative =
        w == pop_weight::kHigh ? cumulative_weight_high : cumulative_weight_low;
    auto const r = rand_unit() * cumulative.back();
    auto const cell = std::upper_bound(cumulative.begin(), cumulative.end(), r);
    auto const cell_idx = std::min(
        static_cast<std::size_t>(std::distance(cumulative.begin(), cell)),
        cumulative.size() - 1U);  // guard against rounding
    return rand_in(cell_stops[cell_idx]);
  };

  auto const ranks = [&] {
    auto ret = std::vector(n, 0U);
    for (auto [i, r] = std::tuple{0U, kMinRank}; i != n;
         ++i, r = r * 2U < master_stops.size() ? r * 2U : kMinRank) {
      ret[i] = r;
    }
    return ret;
  }();

  auto geo_rank_index = 0UL;
  auto const weight_desc = [](pop_weight const w) {
    return w == pop_weight::kHigh
               ? "weighted towards high population (any cell can be drawn, "
                 "likelihood proportional to its population)"
               : "weighted towards low population (any cell can be drawn, "
                 "the k-th least populated cell is as likely as the k-th most "
                 "populated one is for high population)";
  };
  auto const from_desc = population_from
                             ? std::string{weight_desc(*population_from)}
                             : std::string{"drawn uniformly at random"};
  auto to_desc = std::string{};
  if (population_to) {
    to_desc = weight_desc(*population_to);
    lb_rank = false;  // only ever the default here: an explicit 1 was rejected
  } else if (geo_rank) {
    geo_rank_index = 1UL << *geo_rank;
    if (geo_rank_index > master_stops.size() - 1U) {
      fmt::println("geo-rank index exceeds number of stops: {} > {}",
                   geo_rank_index, master_stops.size() - 1U);
      return -1;
    }
    to_desc =
        fmt::format("geo-rank {} of `from`: the {}-th nearest stop by distance",
                    *geo_rank, geo_rank_index);
    lb_rank = false;
  } else if (lb_rank) {
    to_desc =
        "lower-bounds rank of `from`: the 2^n-th stop by travel-time lower "
        "bound, n varied per query";
  } else {
    to_desc = "drawn uniformly at random";
  }
  fmt::println("from: {}", from_desc);
  fmt::println("to:   {}", to_desc);

  auto t = utl::scoped_timer{"generate queries"};
  auto out = std::ofstream{"queries.txt"};
  auto const progress_tracker =
      utl::activate_progress_tracker(fmt::format("generating {} queries", n));
  progress_tracker->in_high(n);
  auto const silencer = utl::global_progress_bars{false};
  auto mutex = std::mutex{};
  struct state {
    api::plan_params p_{};
    std::vector<n::location_idx_t> stops_;
    n::routing::search_state ss_;
    n::routing::raptor_state rs_;
    hash_map<n::location_idx_t, double> geo_distance_;
  };
  utl::parallel_for_run_threadlocal<state>(ranks.size(), [&](state& s,
                                                             auto const i) {
    auto const r = ranks[i];
    s.p_ = master_params;
    s.stops_ = master_stops;
    s.geo_distance_.reserve(master_stops.size());

    auto const get_place =
        [&](n::location_idx_t const l) -> std::optional<std::string> {
      if (!modes) {
        return d.tags_->id(*d.tt_, l);
      }

      auto const nodes =
          node_rtree.in_radius(d.tt_->locations_.coordinates_[l], max_dist);
      if (nodes.empty()) {
        return std::nullopt;
      }

      auto const pos = d.w_->get_node_pos(rand_in(nodes));
      return fmt::format("{},{}", pos.lat(), pos.lng());
    };

    auto const random_from_to = [&] {
      auto from_place = std::optional<std::string>{};
      auto to_place = std::optional<std::string>{};

      for (auto x = 0U; x != kMaxPlaceAttempts; ++x) {
        // stop used to lb-rank the destination (invalid -> random
        // destination)
        auto rank_stop = n::location_idx_t::invalid();
        if (use_flex) {
          auto const seed = rand_in(flex_seeds);
          from_place = fmt::format("{},{}", seed.from_.lat_, seed.from_.lng_);
          rank_stop = seed.rank_stop_;
        } else {
          rank_stop = population_from
                          ? random_population_weighted_stop(*population_from)
                          : random_stop(*d.tt_, s.stops_);
          from_place = get_place(rank_stop);
          if (!from_place) {
            continue;
          }
        }

        if (lb_rank && rank_stop != n::location_idx_t::invalid()) {
          auto const search = n::routing::search<
              n::direction::kBackward,
              n::routing::raptor<n::direction::kBackward, false, 0,
                                 n::routing::search_mode::kOneToAll>>{
              *d.tt_, nullptr, s.ss_, s.rs_,
              nigiri::routing::query{
                  .start_time_ = d.tt_->date_range_.from_,
                  .destination_ = {{rank_stop, n::duration_t{0U}, 0}}}};
          utl::sort(s.stops_, [&](auto const& a, auto const& b) {
            return s.ss_.travel_time_lower_bound_[to_idx(a)] <
                   s.ss_.travel_time_lower_bound_[to_idx(b)];
          });
          to_place = get_place(s.stops_[r]);
        } else if (geo_rank && rank_stop != n::location_idx_t::invalid()) {
          for (auto const l : s.stops_) {
            s.geo_distance_[l] =
                geo::distance(d.tt_->locations_.coordinates_[rank_stop],
                              d.tt_->locations_.coordinates_[l]);
          }
          utl::sort(s.stops_, [&](auto const& a, auto const& b) {
            return s.geo_distance_[a] < s.geo_distance_[b];
          });
          to_place = get_place(s.stops_[geo_rank_index]);
        } else {
          to_place = get_place(
              population_to ? random_population_weighted_stop(*population_to)
                            : random_stop(*d.tt_, s.stops_));
        }
        if (to_place) {
          break;
        }
      }

      utl::verify(from_place.has_value() && to_place.has_value(),
                  "no origin and destination with an eligible OSM node within "
                  "--max_dist {} m found in {} attempts",
                  max_dist, kMaxPlaceAttempts);

      s.p_.fromPlace_ = *from_place;
      s.p_.toPlace_ = *to_place;
    };

    auto const random_time = [&] {
      using namespace std::chrono_literals;
      s.p_.time_ = *first_day +
                   rand_in(0U, static_cast<std::uint32_t>(
                                   (*last_day - *first_day).count())) *
                       date::days{1U} +
                   (time_of_day ? *time_of_day : rand_in(6U, 18U)) * 1h;
    };

    random_from_to();
    random_time();

    auto guard = std::lock_guard{mutex};
    out << s.p_.to_url(kPlanPath) << "\n";
    progress_tracker->increment();
  });

  return 0;
}

}  // namespace motis
