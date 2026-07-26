#include <algorithm>
#include <chrono>
#include <random>

#include "fmt/core.h"

#include "utl/verify.h"

#include "osr/platforms.h"
#include "osr/routing/route.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/get_stops_with_traffic.h"
#include "motis/match_platforms.h"
#include "motis/osr/max_distance.h"
#include "motis/osr/parameters.h"
#include "motis/point_rtree.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace n = nigiri;

namespace motis {

// Benchmarks the 1:N street routing behind `get_offsets()`. Interesting for
// CAR on the destination side: the candidate stops are collected within a
// beeline radius of `max * max_speed`, which the road network never reaches -
// so most targets are unreachable and should be rejected as cheaply as
// possible.
int bench_offsets(int ac, char** av) {
  auto data_path = fs::path{"data"};
  auto mode = std::string{"CAR"};
  auto direction = std::string{"bwd"};
  auto max_seconds = 900U;
  auto max_matching_distance = 250.0;
  auto n_queries = 200U;
  auto seed = 0U;

  auto desc = po::options_description{"Options"};
  add_help_opt(desc);
  add_data_path_opt(desc, data_path);
  desc.add_options()  //
      ("mode,m", po::value(&mode)->default_value(mode),
       "CAR | BIKE | WALK | HGV")  //
      ("dir", po::value(&direction)->default_value(direction),
       "fwd (pre-transit) | bwd (post-transit / destination side)")  //
      ("max", po::value(&max_seconds)->default_value(max_seconds),
       "max street routing duration in seconds")  //
      ("max-matching-distance",
       po::value(&max_matching_distance)->default_value(max_matching_distance),
       "max matching distance in meters")  //
      ("n,n", po::value(&n_queries)->default_value(n_queries),
       "number of queries")  //
      ("seed", po::value(&seed)->default_value(seed), "random seed")  //
      ;

  auto vm = po::variables_map{};
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);
  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const profile = mode == "CAR"     ? osr::search_profile::kCar
                       : mode == "BIKE"  ? osr::search_profile::kBike
                       : mode == "HGV"   ? osr::search_profile::kHgv
                       : mode == "WALK"  ? osr::search_profile::kFoot
                                         : osr::search_profile::kCar;
  auto const dir = direction == "fwd" ? osr::direction::kForward
                                      : osr::direction::kBackward;

  auto const c = config::read(data_path / "config.yml");
  auto d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");
  utl::verify(d.w_ && d.l_ && d.pl_, "street data required");

  auto const& tt = *d.tt_;
  auto const params = to_profile_parameters(profile, {});
  auto const max = std::chrono::seconds{max_seconds};
  auto const max_dist = get_max_distance(profile, {}, max);

  // Draw start locations from the stops that actually have traffic, so the
  // sample matches what the routing endpoint sees.
  auto rng = std::mt19937{seed};
  auto stop_dist = std::uniform_int_distribution<n::location_idx_t::value_t>{
      0U, static_cast<n::location_idx_t::value_t>(tt.n_locations() - 1U)};

  auto targets = std::size_t{0U};
  auto reached = std::size_t{0U};
  auto times_us = std::vector<std::uint64_t>{};

  fmt::println(
      "profile={} dir={} max={}s beeline_radius={:.0f}m "
      "max_matching_distance={}m n={}",
      osr::to_str(profile), direction, max_seconds, max_dist,
      max_matching_distance, n_queries);

  auto n_run = 0U;
  for (auto i = 0U; i != n_queries; ++i) {
    auto const l = n::location_idx_t{stop_dist(rng)};
    auto const pos = osr::location{tt.locations_.coordinates_[l],
                                   d.pl_->get_level(*d.w_, (*d.matches_)[l])};

    auto const near_stops = get_stops_with_traffic(
        tt, nullptr, *d.location_rtree_, pos, max_dist, l);
    if (near_stops.empty()) {
      continue;
    }
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const x) {
          return osr::location{tt.locations_.coordinates_[x],
                               d.pl_->get_level(*d.w_, (*d.matches_)[x])};
        });

    // Matching is shared by all variants and therefore not measured.
    auto const pos_match = d.l_->match(params, pos, false, dir,
                                       max_matching_distance, nullptr, profile);
    auto const near_stop_matches = get_reverse_platform_way_matches(
        *d.l_, d.way_matches_.get(), profile, near_stops, near_stop_locations,
        dir, max_matching_distance);

    auto const start = std::chrono::steady_clock::now();
    auto const paths = osr::route(
        params, *d.w_, *d.l_, profile, pos, near_stop_locations, pos_match,
        near_stop_matches, static_cast<osr::cost_t>(max.count()), dir, nullptr,
        nullptr, d.elevations_.get());
    auto const t = std::chrono::steady_clock::now() - start;

    times_us.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t).count()));
    targets += paths.size();
    reached += static_cast<std::size_t>(
        std::count_if(begin(paths), end(paths),
                      [](auto const& p) { return p.has_value(); }));
    ++n_run;
  }

  utl::verify(n_run != 0U, "no query produced candidate stops");
  fmt::println("queries={} targets/query={}", n_run,
               targets / std::max<std::size_t>(n_run, 1U));

  std::sort(begin(times_us), end(times_us));
  auto const sum =
      std::accumulate(begin(times_us), end(times_us), std::uint64_t{0U});
  auto const q = [&](double const p) {
    return times_us[std::min(times_us.size() - 1U,
                             static_cast<std::size_t>(p * times_us.size()))];
  };
  fmt::println(
      "reached={} ({:.2f}%)  total={:.1f}ms  avg={:.0f}us  "
      "p50={}us  p90={}us  p99={}us  max={}us",
      reached,
      100.0 * static_cast<double>(reached) /
          static_cast<double>(std::max<std::size_t>(targets, 1U)),
      static_cast<double>(sum) / 1000.0,
      static_cast<double>(sum) / static_cast<double>(times_us.size()), q(0.5),
      q(0.9), q(0.99), times_us.back());

  return 0;
}

}  // namespace motis
