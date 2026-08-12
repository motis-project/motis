#include <fstream>
#include <iostream>
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

  auto const parse_population_grid = [&](std::string const& s) {
    auto const file_content = utl::read_file(s.c_str());
    utl::verify(file_content.has_value(),
                "could not read population grid file at {}", s.c_str());
    population_grid = geo::parse_eurostat_population_grid(*file_content);
    utl::erase_if(population_grid, [](auto const c) { return c.data_ == 0UL; });
  };

  auto const grid_cell_rect = [](geo::box const& b) {
    return tg_rect{{b.min_.lng_, b.min_.lat_}, {b.max_.lng_, b.max_.lat_}};
  };

  auto const partial_sum_population_grid = [&] {
    std::partial_sum(
        population_grid.begin(), population_grid.end(), population_grid.begin(),
        [](auto const& acc, auto const& c) -> geo::grid_cell<std::uint64_t> {
          return {c.b_, acc.data_ + c.data_};
        });
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
       "path to a CSV file containing a EUROSTAT population grid; the higher "
       "the population the more likely it is that a grid cell is selected");
  add_data_path_opt(desc, data_path);
  auto vm = parse_opt(ac, av, desc);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

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

  if (!population_grid.empty()) {
    auto const drop_grid_cells_out_of_bounds = [&](tg_geom const* geom,
                                                   char const* label) {
      auto const n_before = population_grid.size();
      auto discarded_population = std::uint64_t{0U};
      utl::erase_if(population_grid, [&](auto const& gc) {
        if (tg_geom_intersects_rect(geom, grid_cell_rect(gc.b_))) {
          return false;
        }
        discarded_population += gc.data_;
        return true;
      });
      auto const n_discarded = n_before - population_grid.size();
      fmt::println(
          "population grid: discarded {}/{} ({:.2f}%) cells outside {}, "
          "discarded population: {}",
          n_discarded, n_before,
          n_before != 0U ? static_cast<double>(n_discarded) /
                               static_cast<double>(n_before) * 100.0
                         : std::numeric_limits<double>::quiet_NaN(),
          label, discarded_population);
      return !population_grid.empty();
    };

    if (bounds != nullptr && !drop_grid_cells_out_of_bounds(bounds, "bounds")) {
      fmt::println(
          "can not generate queries: population grid and bounds are "
          "disjoint");
      return 1;
    }

    if (use_odm_bounds &&
        !drop_grid_cells_out_of_bounds(d.odm_bounds_->geom_, "ODM bounds")) {
      fmt::println(
          "can not generate queries: population grid and ODM bounds are "
          "disjoint");
      return 1;
    }

    auto const total_population = std::accumulate(
        population_grid.begin(), population_grid.end(), std::uint64_t{0U},
        [](auto const acc, auto const& gc) { return acc + gc.data_; });
    fmt::println(
        "population grid: {} cells, total population: {} remaining for "
        "query generation",
        population_grid.size(), total_population);
    partial_sum_population_grid();
  }

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

  auto const ranks = [&] {
    auto ret = std::vector(n, 0U);
    for (auto [i, r] = std::tuple{0U, kMinRank}; i != n;
         ++i, r = r * 2U < master_stops.size() ? r * 2U : kMinRank) {
      ret[i] = r;
    }
    return ret;
  }();

  auto geo_rank_index = 0UL;
  if (geo_rank) {
    fmt::println("from and to pairings by geo-rank = {}", *geo_rank);
    geo_rank_index = 1UL << *geo_rank;
    if (geo_rank_index > master_stops.size() - 1U) {
      fmt::println("geo-rank index exceeds number of stops: {} > {}",
                   geo_rank_index, master_stops.size() - 1U);
      return -1;
    }
    lb_rank = false;
  } else if (lb_rank) {
    fmt::println("from and to pairings by lower bounds rank");
  } else {
    fmt::println("from and to uniformly at random");
  }

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

      for (auto x = 0U; x != 1000U; ++x) {
        // stop used to lb-rank the destination (invalid -> random
        // destination)
        auto rank_stop = n::location_idx_t::invalid();
        if (use_flex) {
          auto const seed = rand_in(flex_seeds);
          from_place = fmt::format("{},{}", seed.from_.lat_, seed.from_.lng_);
          rank_stop = seed.rank_stop_;
        } else {
          rank_stop = random_stop(*d.tt_, s.stops_);
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
          to_place = get_place(random_stop(*d.tt_, s.stops_));
        }
        if (to_place) {
          break;
        }
      }

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
