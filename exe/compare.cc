#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

#include "conf/configuration.h"

#include "boost/json/parse.hpp"
#include "boost/json/serialize.hpp"
#include "boost/json/value_from.hpp"
#include "boost/json/value_to.hpp"

#include "fmt/std.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/sorted_diff.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis-api/motis-api.h"
#include "motis/types.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace motis {

int compare(int ac, char** av) {
  auto queries_path = fs::path{"queries.txt"};
  auto responses_paths = std::vector<std::string>{};
  auto fails_path = fs::path{"fail"};
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

  auto const write_fails = fs::is_directory(fails_path);
  if (!write_fails) {
    fmt::println("{} is not a directory, not writing fails", fails_path);
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
    if (f.eof() || f.peek() == EOF) {
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
    return x.params_.has_value() && !x.responses_.empty() &&
           utl::all_of(x.responses_, [](auto&& r) { return r.has_value(); });
  };
  auto const params = [](api::Itinerary const& x) {
    return std::tie(x.startTime_, x.endTime_, x.transfers_);
  };
  auto const print_params = [](api::Itinerary const& x) {
    std::cout << x.startTime_ << ", " << x.endTime_
              << ", transfers=" << std::setw(2) << std::left << x.transfers_;
  };
  auto const print_none = []() { std::cout << "\t\t\t\t\t\t"; };
  auto n_equal = 0U;
  auto const print_differences = [&](info const& x) {
    auto const& ref = x.responses_[0].value().itineraries_;
    auto mismatch = false;
    for (auto i = 1U; i < x.responses_.size(); ++i) {
      auto const uut = x.responses_[i].value().itineraries_;
      if (std::ranges::equal(ref | std::views::transform(params),
                             uut | std::views::transform(params))) {
        ++n_equal;
        continue;
      }

      mismatch = true;
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
                  std::cout << "\t\t\t\t";
                  print_params(j);
                  std::cout << "\n";
                } else {
                  print_params(j);
                  std::cout << "\t\t\t\t";
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

    if (mismatch && write_fails) {
      std::ofstream{fails_path / fmt::format("{}_q.txt", x.id_)}
          << x.params_->to_url("/v1/plan");
      for (auto i = 0U; i < x.responses_.size(); ++i) {
        std::ofstream{fails_path / fmt::format("{}_{}.json", x.id_, i)}
            << boost::json::serialize(
                   boost::json::value_from(x.responses_[i].value()));
      }
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