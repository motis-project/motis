#include "conf/configuration.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <ranges>

#include "boost/json/parse.hpp"
#include "boost/json/value_to.hpp"
#include "boost/url/url.hpp"

#include "fmt/std.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/sorted_diff.h"
#include "utl/timing.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/tag_lookup.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {
int compare(int ac, char** av) {
  auto queries_path = fs::path{"queries.txt"};
  auto responses_paths = std::vector<std::string>{};
  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("queries,q", po::value(&queries_path)->default_value(queries_path),
       "queries file")  //
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

  struct info {
    unsigned id_;
    std::optional<api::plan_params> params_{};
    std::vector<std::optional<api::plan_response>> responses_{};
  };
  auto response_buf = hash_map<unsigned, info>{};
  auto const get = [&](unsigned const id) -> info& {
    return utl::get_or_create(response_buf, id, [&]() {
      auto x = info{.id_ = id};
      x.responses_.resize(responses_paths.size());
      return x;
    });
  };
  auto const is_finished = [](info const& x) {
    return x.params_.has_value() &&
           utl::all_of(x.responses_, [](auto&& r) { return r.has_value(); });
  };
  struct intermodal {
    std::string to_str() const {
      auto ss = std::stringstream{};
      ss << mode_ << " " << std::chrono::round<std::chrono::minutes>(duration_);
      return ss.str();
    }

    bool operator<(intermodal const& b) const {
      return std::tie(mode_, duration_) < std::tie(b.mode_, b.duration_);
    }

    bool operator==(intermodal const& b) const {
      return std::tie(mode_, duration_) == std::tie(b.mode_, b.duration_);
    }

    api::ModeEnum mode_;
    std::chrono::seconds duration_;
  };

  auto const params = [](api::Itinerary const& x) {
    auto const pre_transit = [&]() -> std::optional<intermodal> {
      if (x.legs_.front().mode_ != api::ModeEnum::TRANSIT) {
        return intermodal{x.legs_.front().mode_,
                          std::chrono::seconds{x.legs_.front().duration_}};
      } else {
        return std::nullopt;
      }
    };
    auto const post_transit = [&]() -> std::optional<intermodal> {
      if (x.legs_.back().mode_ != api::ModeEnum::TRANSIT) {
        return intermodal{x.legs_.back().mode_,
                          std::chrono::seconds{x.legs_.back().duration_}};
      } else {
        return std::nullopt;
      }
    };
    return std::tuple(x.startTime_, x.endTime_, x.transfers_, pre_transit(),
                      post_transit());
  };
  auto const print_params = [&](api::Itinerary const& x) {
    auto const time = [](auto const& t) {
      return std::format("{0:%F}T{0:%R}", t);
    };
    auto const p = params(x);
    std::cout << time(std::get<0>(p).time_) << ", "
              << time(std::get<1>(p).time_) << ", " << std::get<2>(p) << ", "
              << (std::get<3>(p) ? std::get<3>(p).value().to_str() : "NULL")
              << ", "
              << (std::get<4>(p) ? std::get<4>(p).value().to_str() : "NULL");
  };
  auto const print_none = []() { std::cout << "\t\t\t\t\t\t"; };
  auto n_equal = 0U;
  auto const print_differences = [&](info const& x) {
    auto const& ref = x.responses_[0].value().itineraries_;
    for (auto i = 1U; i < x.responses_.size(); ++i) {
      auto const uut = x.responses_[i].value().itineraries_;
      if (std::ranges::equal(ref | std::views::transform(params),
                             uut | std::views::transform(params))) {
        ++n_equal;
        continue;
      }

      std::cout << "QUERY=" << x.id_ << "\n";
      utl::sorted_diff(
          ref, uut,
          [&](api::Itinerary const& a, api::Itinerary const& b) {
            return params(a) < params(b);
          },
          [&](api::Itinerary const&, api::Itinerary const&) {
            return false;  // always call for equal
          },
          utl::overloaded{
              [&](utl::op op, api::Itinerary const& j) {
                if (op == utl::op::kAdd) {
                  print_none();
                  std::cout << "\t\t\t";
                  print_params(j);
                  std::cout << "\n";
                } else {
                  print_params(j);
                  std::cout << "\t\t\t";
                  print_none();
                  std::cout << "\n";
                }
              },
              [&](api::Itinerary const& a, api::Itinerary const& b) {
                print_params(a);
                std::cout << "\t\t\t";
                print_params(b);
                std::cout << "\n";
              }});
      std::cout << "\n\n";
    }
  };
  auto n_consumed = 0U;
  auto const consume_if_finished = [&](info const& x) {
    if (!is_finished(x)) {
      return;
    }
    print_differences(x);
    response_buf.erase(x.id_);
    ++n_consumed;
  };

  auto query_file = open_file(queries_path);
  auto responses_files =
      utl::to_vec(responses_paths, [&](auto&& p) { return open_file(p); });

  auto query_id = 0U;
  auto done = false;
  while (!done) {
    done = true;

    if (auto const q = read_line(query_file); q.has_value()) {
      std::cout << "query_id: " << query_id << "\n";
      auto& info = get(query_id++);
      info.params_ = api::plan_params{boost::urls::url{*q}.params()};
      consume_if_finished(info);
      done = false;
    }

    for (auto const [i, res_file] : utl::enumerate(responses_files)) {

      if (auto const r = read_line(res_file); r.has_value()) {
        auto res =
            boost::json::value_to<api::plan_response>(boost::json::parse(*r));
        utl::sort(res.itineraries_,
                  [&](auto&& a, auto&& b) { return params(a) < params(b); });
        auto const id = res.debugOutput_.at("id");
        auto& info = get(static_cast<unsigned>(id));
        info.responses_[i] = std::move(res);
        consume_if_finished(info);
        done = false;
      }
    }
  }

  std::cout << "consumed: " << n_consumed << "\n";
  std::cout << "buffered: " << response_buf.size() << "\n";
  std::cout << "   equal: " << n_equal << "\n";

  return 0;
}

}  // namespace motis