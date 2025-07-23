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
  auto modes = std::optional<std::vector<api::ModeEnum>>{};
  auto max_dist = 800.0;  // m
  auto use_walk = false;
  auto use_bike = false;
  auto use_car = false;
  auto use_odm = false;

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

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("n,n", po::value(&n)->default_value(n), "number of queries")  //
      ("first_day", po::value<std::string>()->notifier(parse_first_day),
       "first day of query generation, format: YYYY-MM-DD")  //
      ("last_day", po::value<std::string>()->notifier(parse_last_day),
       "last day of query generation, format: YYYY-MM-DD")  //
      ("modes,m", po::value<std::string>()->notifier(parse_modes),
       "comma-separated list of modes for first/last mile and direct (requires "
       "street routing), supported: WALK, BIKE, CAR, ODM")  //
      ("max_dist", po::value(&max_dist)->default_value(max_dist),
       "maximum distance from a public transit stop in meters, only used for "
       "intermodal queries");
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

    for (auto i = osr::node_idx_t{0U}; i < d.w_->n_nodes(); ++i) {
      auto const& props = d.w_->r_->node_properties_[i];
      if ((props.is_walk_accessible() && use_walk) ||
          (props.is_bike_accessible() && use_bike) ||
          (props.is_car_accessible() && (use_car || use_odm))) {
        if (auto const& pos = d.w_->get_node_pos(i);
            !use_odm_bounds || d.odm_bounds_->contains(pos)) {
          node_rtree.add(pos, i);
        }
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
      auto p = api::plan_params{};
      using namespace std::chrono_literals;
      p.fromPlace_ = random_place();
      p.toPlace_ = random_place();
      p.time_ = *first_day +
                rand_in(0U, static_cast<std::uint32_t>(
                                (*last_day - *first_day).count())) *
                    date::days{1U} +
                rand_in(6U, 18U) * 1h;
      if (modes) {
        p.directModes_ = *modes;
        p.preTransitModes_ = *modes;
        p.postTransitModes_ = *modes;
      }
      out << p.to_url("/api/v1/plan") << "\n";
    }
  }

  return 0;
}

}  // namespace motis
