#include "conf/configuration.h"

#include <random>

#include "boost/url/url.hpp"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/point_rtree.h"

#include "./flags.h"
#include "motis/place.h"

namespace n = nigiri;
namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace motis {

auto& rng() {
  static auto rng = std::mt19937{};
  return rng;
}

template <typename A, typename B>
std::common_type_t<A, B> rand_in(A const start, B const end) {
  using T = std::common_type_t<A, B>;
  auto dist =
      std::uniform_int_distribution{static_cast<T>(start), static_cast<T>(end)};
  return dist(rng());
}

template <typename It>
It rand_in(It const begin, It const end) {
  return std::next(begin, rand_in(0, std::distance(begin, end) - 1));
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
  auto n = 1000U;

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("n", po::value(&n)->default_value(n), "number of queries");
  add_data_path_opt(desc, data_path);
  auto vm = parse_opt(ac, av, desc);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const c = config::read(data_path / "config.yml");
  if (!c.timetable_.has_value()) {
    std::cout << "Timetable required but not set\n";
    return 1;
  }

  auto const d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");

  auto stops = std::vector<n::location_idx_t>{};
  stops.resize(d.tt_->n_locations());
  for (auto i = 0U; i != stops.size(); ++i) {
    stops[i] = n::location_idx_t{i};
  }

  auto const p = api::plan_params{};
  for (auto i = 0U; i != n; ++i) {
    p.fromPlace_ = to_place(random_stop(*d.tt_, stops));
  }

  auto u = boost::urls::url{"/"};
  u.params().append({"A", "B"});
  u.params().append({"A", "B"});
  std::cout << u << "\n";

  return 0;
}

}  // namespace motis