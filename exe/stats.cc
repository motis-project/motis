#include "conf/configuration.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <ranges>

#include "boost/json/parse.hpp"
#include "boost/json/value_to.hpp"

#include "cista/reflection/for_each_field.h"

#include "fmt/std.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/timing.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {
int stats(int ac, char** av) {
  auto responses_paths = std::vector<std::string>{};
  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("responses,r",
       po::value(&responses_paths)
           ->multitoken()
           ->default_value(responses_paths),
       "response files");

  auto vm = parse_opt(ac, av, desc);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const open_file = [](fs::path const& p) {
    auto f = std::ifstream{};
    f.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    try {
      f.open(p);
    } catch (std::exception const& e) {
      throw utl::fail("could not open file \"{}\": {}", p, e.what());
    }
    return f;
  };
  auto const read_line = [](std::ifstream& f) -> std::optional<std::string> {
    if (f.peek() == EOF || f.eof()) {
      return std::nullopt;
    }
    std::string line;
    std::getline(f, line);
    return line;
  };
  auto responses_files =
      utl::to_vec(responses_paths, [&](auto&& p) { return open_file(p); });
  std::vector<std::vector<std::chrono::milliseconds>> wall_time{
      responses_paths.size()};
  std::vector<std::vector<std::chrono::minutes>> travel_time{
      responses_paths.size()};
  std::vector<std::vector<std::uint8_t>> transfers{responses_paths.size()};
  std::vector<std::vector<std::chrono::minutes>> walk_time{
      responses_paths.size()};
  std::vector<std::vector<std::chrono::minutes>> bike_time{
      responses_paths.size()};
  std::vector<std::vector<std::chrono::minutes>> car_time{
      responses_paths.size()};
  for (auto const [i, res_file] : utl::enumerate(responses_files)) {
    fmt::println("({}) {}", i, responses_paths[i]);
    while (auto const r = read_line(res_file)) {
      auto res =
          boost::json::value_to<api::plan_response>(boost::json::parse(*r));
      wall_time[i].emplace_back(res.debugOutput_["execute_time"]);
      for (auto const& j : res.itineraries_) {
        travel_time[i].emplace_back(std::chrono::round<std::chrono::minutes>(
            std::chrono::seconds{j.duration_}));
        transfers[i].emplace_back(j.transfers_);
        auto const add_intermodal = [&](auto const& l, auto const k) {
          switch (l.mode_) {
            case api::ModeEnum::WALK:
              walk_time[k].emplace_back(
                  std::chrono::round<std::chrono::minutes>(
                      std::chrono::seconds{l.duration_}));
              break;
            case api::ModeEnum::BIKE:
              bike_time[k].emplace_back(
                  std::chrono::round<std::chrono::minutes>(
                      std::chrono::seconds{l.duration_}));
              break;
            case api::ModeEnum::CAR:
              car_time[k].emplace_back(std::chrono::round<std::chrono::minutes>(
                  std::chrono::seconds{l.duration_}));
              break;
            default: break;
          }
        };
        add_intermodal(j.legs_.front(), i);
        add_intermodal(j.legs_.back(), i);
      }
    }
  }

  auto const sort_all = [](auto& attr) {
    for (auto& vals : attr) {
      utl::sort(vals);
    }
  };
  sort_all(wall_time);
  sort_all(travel_time);
  sort_all(transfers);
  sort_all(walk_time);
  sort_all(bike_time);
  sort_all(car_time);

  auto const cols =
      std::vector<std::string>{"25%", "50%", "75%", "99%", "99.9%", "100%"};
  auto const quantiles = std::vector<double>{0.25, 0.5, 0.75, 0.99, 0.999, 1.0};
  auto const print_header = [&]() {
    std::cout << "   ";
    for (const auto& c : cols) {
      std::cout << std::format("{: >{}}", c, 12);
    }
    std::cout << "\n";
  };
  auto const get_quantile = [](auto const& v, double q) {
    q = q < 0.0 ? 0.0 : q;
    q = 1.0 < q ? 1.0 : q;
    if (q == 1.0) {
      return v.back();
    }
    return v[std::floor(static_cast<double>(v.size()) * q)];
  };
  auto const print_quantiles = [&](auto const& v) {
    for (auto const q : quantiles) {
      std::cout << std::format("{:>{}}", get_quantile(v, q), 12);
    }
    std::cout << "\n";
  };
  auto const print_attr = [&](auto const& attr, std::string_view const name) {
    fmt::println("\n{}", name);
    print_header();
    for (auto const [i, vals] : utl::enumerate(attr)) {
      std::cout << std::format("({})", i);
      if (vals.empty()) {
        std::cout << "\n";
      } else {
        print_quantiles(vals);
      }
    }
  };
  print_attr(wall_time, "Wall Time");
  print_attr(travel_time, "Travel Time");
  print_attr(transfers, "Transfers");
  print_attr(walk_time, "Walk Time");
  print_attr(bike_time, "Bike Time");
  print_attr(car_time, "Car Time");

  return 0;
}

}  // namespace motis