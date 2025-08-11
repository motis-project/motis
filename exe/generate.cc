#include "boost/format/format_fwd.hpp"

#include "conf/configuration.h"

#include <fstream>
#include <iostream>
#include <random>
#include <ranges>

#include "boost/json/serialize.hpp"
#include "boost/url/url.hpp"

#include "fmt/std.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/odm/bounds.h"
#include "motis/point_rtree.h"
#include "motis/tag_lookup.h"

#include "./flags.h"

namespace n = nigiri;
namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {

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
  auto p = api::plan_params{};

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
  };

  auto const parse_time_of_day = [&](std::uint32_t const h) {
    time_of_day = h % 24U;
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
           [&](auto const v) { p.maxTravelTime_ = v; }),
       "sets maximum travel time of the queries")  //
      ("max_matching_distance",
       po::value(&p.maxMatchingDistance_)
           ->default_value(p.maxMatchingDistance_),
       "sets the maximum matching distance of the queries")  //
      ("fastest_direct_factor",
       po::value(&p.fastestDirectFactor_)
           ->default_value(p.fastestDirectFactor_),
       "sets fastest direct factor of the queries");
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

  first_day = first_day ? d.tt_->date_range_.clamp(*first_day)
                        : d.tt_->date_range_.from_;
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
  fmt::println("date range: [{}, {}]", *first_day, *last_day);

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

    p.directModes_ = *modes;
    p.preTransitModes_ = *modes;
    p.postTransitModes_ = *modes;

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

    auto const in_bounds = [&](auto const& pos) {
      return !use_odm_bounds || d.odm_bounds_->contains(pos);
    };

    for (auto i = osr::node_idx_t{0U}; i < d.w_->n_nodes(); ++i) {
      if (mode_match(i) && in_bounds(d.w_->get_node_pos(i))) {
        node_rtree.add(d.w_->get_node_pos(i), i);
      }
    }
  }

  auto stops = std::vector<n::location_idx_t>{};
  for (auto i = 0U; i != d.tt_->n_locations(); ++i) {
    auto const l = n::location_idx_t{i};
    if (use_odm_bounds &&
        !d.odm_bounds_->contains(d.tt_->locations_.coordinates_[l])) {
      continue;
    }
    stops.emplace_back(l);
  }

  auto const random_place = [&]() {
    if (!modes) {
      return d.tags_->id(*d.tt_, random_stop(*d.tt_, stops));
    }

    auto nodes = std::vector<osr::node_idx_t>{};
    do {
      nodes = node_rtree.in_radius(
          d.tt_->locations_.coordinates_[random_stop(*d.tt_, stops)], max_dist);
    } while (nodes.empty());

    auto pos = d.w_->get_node_pos(rand_in(nodes));
    return fmt::format("{},{}", pos.lat(), pos.lng());
  };

  {
    auto out = std::ofstream{"queries.txt"};
    for (auto i = 0U; i != n; ++i) {
      using namespace std::chrono_literals;
      p.fromPlace_ = random_place();
      p.toPlace_ = random_place();
      p.time_ = *first_day +
                rand_in(0U, static_cast<std::uint32_t>(
                                (*last_day - *first_day).count())) *
                    date::days{1U} +
                (time_of_day ? *time_of_day : rand_in(6U, 18U)) * 1h;
      out << p.to_url("/api/v1/plan") << "\n";
    }
  }

  return 0;
}

}  // namespace motis
