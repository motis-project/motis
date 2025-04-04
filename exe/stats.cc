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

  struct stats {
    std::vector<std::chrono::milliseconds> wall_time_{};
    std::vector<std::chrono::minutes> travel_time_{};
    std::vector<std::uint8_t> transfers_{};
    std::vector<std::chrono::minutes> walk_time_{};
    std::vector<std::chrono::minutes> bike_time_{};
    std::vector<std::chrono::minutes> car_time_{};
  };

  auto s = std::vector<stats>{responses_paths.size()};
  for (auto const [i, res_file] : utl::enumerate(responses_files)) {
    fmt::println("({}) {}", i, responses_paths[i]);
    while (auto const r = read_line(res_file)) {
      auto res =
          boost::json::value_to<api::plan_response>(boost::json::parse(*r));
      s[i].wall_time_.emplace_back(res.debugOutput_["execute_time"]);
      for (auto const& j : res.itineraries_) {
        s[i].travel_time_.emplace_back(std::chrono::round<std::chrono::minutes>(
            std::chrono::seconds{j.duration_}));
        s[i].transfers_.emplace_back(j.transfers_);
        switch (j.legs_.front().mode_) {
          case api::ModeEnum::WALK:
            s[i].walk_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.front().duration_}));
            break;
          case api::ModeEnum::BIKE:
            s[i].bike_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.front().duration_}));
            break;
          case api::ModeEnum::CAR:
            s[i].car_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.front().duration_}));
            break;
          default: break;
        }
        switch (j.legs_.back().mode_) {
          case api::ModeEnum::WALK:
            s[i].walk_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.back().duration_}));
            break;
          case api::ModeEnum::BIKE:
            s[i].bike_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.back().duration_}));
            break;
          case api::ModeEnum::CAR:
            s[i].car_time_.emplace_back(
                std::chrono::round<std::chrono::minutes>(
                    std::chrono::seconds{j.legs_.back().duration_}));
            break;
          default: break;
        }
      }
    }
  }

  for (auto& u : s) {
    utl::sort(u.wall_time_);
    utl::sort(u.travel_time_);
    utl::sort(u.transfers_);
    utl::sort(u.walk_time_);
    utl::sort(u.bike_time_);
    utl::sort(u.car_time_);
  }
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

  fmt::println("\nWall Time");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.wall_time_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.wall_time_);
    }
  }

  fmt::println("\nTravel Time");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.travel_time_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.travel_time_);
    }
  }

  fmt::println("\nTransfers");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.transfers_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.transfers_);
    }
  }

  fmt::println("\nWalk Time");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.walk_time_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.walk_time_);
    }
  }

  fmt::println("\nBike Time");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.bike_time_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.bike_time_);
    }
  }

  fmt::println("\nCar Time");
  print_header();
  for (auto const [i, u] : utl::enumerate(s)) {
    std::cout << std::format("({})", i);
    if (u.car_time_.empty()) {
      std::cout << "\n";
    } else {
      print_quantiles(u.car_time_);
    }
  }

  return 0;
}

}  // namespace motis