#include "motis/routing/eval/commands.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "utl/get_or_create.h"

#include "motis/module/message.h"
#include "motis/protocol/RoutingResponse_generated.h"
#include "motis/routing/allocator.h"

using namespace motis;
using namespace motis::module;
using namespace motis::routing;

namespace po = boost::program_options;

namespace motis::routing::eval {

struct thousands_sep : std::numpunct<char> {
  char_type do_thousands_sep() const override { return ','; }
  string_type do_grouping() const override { return "\3"; }
};

struct stat {
  struct entry {
    bool operator<(entry const& o) const { return value_ < o.value_; }
    uint64_t msg_id_, value_;
  };

  stat() = default;
  stat(std::string name, uint64_t count_so_far)
      : name_(std::move(name)), values_(count_so_far) {}

  void add(uint64_t msg_id, uint64_t value) {
    values_.emplace_back(entry{msg_id, value});
    sum_ += value;
  }

  std::string name_;
  std::vector<entry> values_;
  uint64_t sum_{};
};

struct category {
  category() = default;
  explicit category(std::string name) : name_(std::move(name)) {}

  std::string name_;
  std::map<std::string, stat> stats_;
};

stat::entry quantile(std::vector<stat::entry> const& sorted_values, double q) {
  if (q == 1.0) {
    return sorted_values.back();
  } else {
    return sorted_values[std::min(
        static_cast<std::size_t>(std::round(q * (sorted_values.size() - 1))),
        sorted_values.size() - 1)];
  }
}

void print_category(category& cat, uint64_t count, bool compact, int top) {
  std::cout << "\n"
            << cat.name_ << "\n"
            << std::string(cat.name_.size(), '=') << "\n"
            << std::endl;
  for (auto& s : cat.stats_) {
    auto& stat = s.second;
    if (stat.values_.empty()) {
      continue;
    }
    std::sort(begin(stat.values_), end(stat.values_));
    auto const avg = (stat.sum_ / static_cast<double>(count));
    if (compact) {
      std::cout << std::left << std::setw(30) << stat.name_
                << " avg: " << std::setw(27) << std::setprecision(4)
                << std::fixed << avg << " Q(99): " << std::setw(25)
                << quantile(stat.values_, 0.99).value_
                << " Q(90): " << std::setw(22)
                << quantile(stat.values_, 0.9).value_
                << " Q(80): " << std::setw(22)
                << quantile(stat.values_, 0.8).value_
                << " Q(50): " << std::setw(22)
                << quantile(stat.values_, 0.5).value_;

      auto const from = std::max(
          uint64_t{}, static_cast<uint64_t>(stat.values_.size()) - top);
      for (int i = from; i != stat.values_.size(); ++i) {
        auto const i_rev = stat.values_.size() - (i - from) - 1;
        std::cout << "(v=" << stat.values_[i_rev].value_
                  << ", i=" << stat.values_[i_rev].msg_id_ << ")";
        if (i != stat.values_.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << std::endl;
    } else {
      std::cout
          << stat.name_ << "\n      average: " << std::setprecision(4)
          << std::fixed << avg << "\n          max: "
          << std::max_element(begin(stat.values_), end(stat.values_))->value_
          << "\n  99 quantile: " << quantile(stat.values_, 0.99).value_
          << "\n  90 quantile: " << quantile(stat.values_, 0.9).value_
          << "\n  80 quantile: " << quantile(stat.values_, 0.8).value_
          << "\n  50 quantile: " << quantile(stat.values_, 0.5).value_
          << "\n          min: "
          << std::min_element(begin(stat.values_), end(stat.values_))->value_
          << "\n"
          << std::endl;
    }
  }
}

int analyze_results(int argc, char const** argv) {
  bool help = false;
  bool include_empty = false;
  bool long_output = false;
  int top = 0;
  std::string filename = "results.txt";
  std::vector<std::string> filtered_categories;

  po::options_description desc("Options:");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help), "show help")
      ("long,l", po::bool_switch(&long_output), "long output")
      ("include-empty,e", po::bool_switch(&include_empty),
       "include results without connections")
      ("file,f", po::value<std::string>(&filename)->default_value(filename),
       "results.txt filename")
      ("category,c", po::value<std::vector<std::string>>(&filtered_categories),
       "only print selected categories")
      ("top,t", po::value<int>(&top), "display the top N values")
      ;
  // clang-format on
  po::positional_options_description pod;
  pod.add("file", 1);
  pod.add("category", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(pod).run(),
      vm);
  po::notify(vm);

  if (help) {
    std::cout << "Usage: " << argv[0]
              << " [options] {results.txt} [{category}...]\n"
              << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));

  std::map<std::string, category> categories;
  stat connection_count;

  std::ifstream in(filename);
  std::string line;

  uint64_t total_count = 0;
  uint64_t count = 0;
  uint64_t no_con_count = 0;
  uint64_t invalid = 0;

  while (in.peek() != EOF && !in.eof()) {
    std::getline(in, line);

    auto const res_msg = make_msg(line);
    if (res_msg->get()->content_type() != MsgContent_RoutingResponse) {
      ++invalid;
      continue;
    }
    auto const res = motis_content(RoutingResponse, res_msg);

    const auto cc = res->connections()->size();
    connection_count.add(res_msg->id(), cc);
    ++total_count;

    if (cc == 0) {
      ++no_con_count;
      if (!include_empty) {
        continue;
      }
    }

    for (auto const& rc : *res->statistics()) {
      auto& cat = utl::get_or_create(categories, rc->category()->str(), [&]() {
        return category{rc->category()->str()};
      });
      for (auto const& rs : *rc->entries()) {
        auto& s = utl::get_or_create(cat.stats_, rs->name()->str(), [&]() {
          return stat(rs->name()->str(), count);
        });
        s.add(res_msg->id(), rs->value());
      }
    }

    for (auto& c : categories) {
      auto const rc = res->statistics()->LookupByKey(c.first.c_str());
      if (rc == nullptr) {
        for (auto& s : c.second.stats_) {
          s.second.add(res_msg->id(), 0);
        }
      } else {
        for (auto& s : c.second.stats_) {
          auto const rs = rc->entries()->LookupByKey(s.first.c_str());
          if (rs == nullptr) {
            s.second.add(res_msg->id(), 0);
          }
        }
      }
    }

    ++count;
  }

  if (categories.empty()) {
    std::cout << "no stats found\n";
    return 0;
  }

  std::cout << "   total count: " << total_count << "\n"
            << "       no conn: " << no_con_count << "\n"
            << "        errors: " << invalid << "\n"
            << "       counted: " << count << "\n"
            << std::endl;

  if (filtered_categories.empty()) {
    for (auto& c : categories) {
      print_category(c.second, count, !long_output, top);
    }
  } else {
    for (auto const& name : filtered_categories) {
      auto const c = categories.find(name);
      if (c != end(categories)) {
        print_category(c->second, count, !long_output, top);
      } else {
        std::cout << "\n\n" << name << ": not found\n\n";
      }
    }
  }

  return 0;
}

}  // namespace motis::routing::eval