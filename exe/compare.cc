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
#include "utl/file_utils.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/sorted_diff.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis-api/motis-api.h"
#include "motis/constants.h"
#include "motis/qa.h"
#include "motis/types.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {

int compare(int ac, char** av) {
  auto subset_check = false;
  auto queries_path = fs::path{"queries.txt"};
  auto responses_paths = std::vector<std::string>{};
  auto fails_path = fs::path{"fail"};
  auto walk = false;
  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("queries,q", po::value(&queries_path)->default_value(queries_path),
       "queries file")  //
      ("subset_check", po::value(&subset_check)->default_value(subset_check),
       "only check subset ([1...N] <= [0])")  //
      ("responses,r",
       po::value(&responses_paths)
           ->multitoken()
           ->default_value(responses_paths),
       "response files")  //
      ("walking_time,w", po::bool_switch(&walk)->default_value(walk),
       "whether walking time should be considered, default: off");

  auto vm = parse_opt(ac, av, desc);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const write_fails = fs::is_directory(fails_path);
  if (!write_fails) {
    fmt::println("{} is not a directory, not writing fails", fails_path);
  }

  struct info {
    unsigned id_;
    std::optional<api::plan_params> params_{};
    std::vector<std::optional<api::plan_response>> responses_{};
  };
  auto const params = walk ? [](api::Itinerary const& x) {
    return std::tuple(x.startTime_, x.endTime_, x.transfers_,
                      qa::criterion::kDefaultWalkingTime(x));
  } : [](api::Itinerary const& x) {
    return std::tuple(x.startTime_, x.endTime_, x.transfers_, 0.0);
  };
  auto const equal = [&](std::vector<api::Itinerary> const& a,
                         std::vector<api::Itinerary> const& b) {
    if (subset_check) {
      return utl::all_of(a, [&](api::Itinerary const& x) {
        return utl::any_of(
            b, [&](api::Itinerary const& y) { return params(x) == params(y); });
      });
    } else {
      return std::ranges::equal(a | std::views::transform(params),
                                b | std::views::transform(params));
    }
  };
  auto const print_params = [](api::Itinerary const& x) {
    std::cout << x.startTime_ << ", " << x.endTime_
              << ", transfers=" << std::setw(2) << std::right << x.transfers_
              << ", walk=" << std::setw(4) << std::right
              << qa::criterion::kDefaultWalkingTime(x);
  };
  auto const get_ratings = [](auto const& ref, auto const& uut) {
    return std::map<std::string, double>{
        {"start_end_transfer", qa::rate(ref, uut, qa::kStartEndTransfer)},
        {"start_end_transfer_walk",
         qa::rate(ref, uut, qa::kStartEndTransferWalk)}};
  };
  auto const print_ratings = [](auto const& ratings) {
    for (auto const& [name, rating] : ratings) {
      std::cout << "[" << name << ", " << rating << ", "
                << (rating > 0   ? "↗"
                    : rating < 0 ? "↘"
                                 : "→")
                << " ] ";
    }
  };
  auto const print_none = []() { std::cout << "\t\t\t\t\t\t"; };
  auto n_equal = 0U;
  auto const print_differences = [&](info const& x) {
    auto const is_incomplete =
        utl::any_of(x.responses_, [](auto&& x) { return !x.has_value(); });

    auto const ref =
        x.responses_[0].value_or(api::plan_response{}).itineraries_;
    auto mismatch = false;
    for (auto i = 1U; i < x.responses_.size(); ++i) {
      mismatch |= !x.responses_[i].has_value();

      auto const uut =
          x.responses_[i].value_or(api::plan_response{}).itineraries_;
      if (equal(ref, uut)) {
        ++n_equal;
        continue;
      }

      mismatch = true;
      std::cout << "QUERY=" << x.id_ << " [" << x.params_->to_url(kPlanPath)
                << "]";
      if (is_incomplete) {
        std::cout << " [INCOMPLETE!!]";
      }
      std::cout << "\n";
      if (x.responses_.size() == 2U) {
        std::cout << "ratings: ";
        print_ratings(get_ratings(ref, uut));
        std::cout << "\n";
      }
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
                std::cout << "\t";
                print_params(b);
                std::cout << "\n";
              }});
      std::cout << "\n\n";
    }

    if (mismatch && write_fails) {
      std::ofstream{fails_path / fmt::format("{}_q.txt", x.id_)}
          << x.params_->to_url(kPlanPath) << "\n";
      for (auto i = 0U; i < x.responses_.size(); ++i) {
        if (!x.responses_[i].has_value()) {
          continue;
        }
        std::ofstream{fails_path / fmt::format("{}_{}.json", x.id_, i)}
            << json::serialize(json::value_from(x.responses_[i].value()))
            << "\n";
      }
    }
  };

  auto query_file = utl::open_file(queries_path);
  auto responses_files =
      utl::to_vec(responses_paths, [&](auto&& p) { return utl::open_file(p); });

  auto n_consumed = 0U;
  auto query_id = 0U;
  while (true) {
    auto nfo =
        info{.id_ = ++query_id,
             .responses_ = std::vector<std::optional<api::plan_response>>{
                 responses_files.size()}};

    if (auto const q = utl::read_line(query_file); q.has_value()) {
      nfo.params_ = api::plan_params{boost::urls::url{*q}.params()};
    } else {
      break;
    }

    for (auto const [i, res_file] : utl::enumerate(responses_files)) {
      if (auto const r = utl::read_line(res_file); r.has_value()) {
        try {
          auto val = boost::json::parse(*r);
          if (val.is_object() &&
              val.as_object().contains("requestParameters")) {
            auto res = json::value_to<api::plan_response>(val);
            utl::sort(res.itineraries_, [&](auto&& a, auto&& b) {
              return params(a) < params(b);
            });
            nfo.responses_[i] = std::move(res);
          }
        } catch (...) {
          fmt::println(
              "[WARNING] query_id: {}, response: {}, could not parse: {}",
              query_id, responses_paths[i], *r);
        }
      } else {
        break;
      }
    }

    print_differences(nfo);
    ++n_consumed;
  }

  std::cout << "consumed: " << n_consumed << "\n";
  std::cout << "   equal: " << n_equal << "\n";

  return n_consumed == n_equal ? 0 : 1;
}

}  // namespace motis