#include <fstream>
#include <iostream>

#include "conf/configuration.h"

#include "utl/init_from.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/motis_instance.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

struct thousands_sep : std::numpunct<char> {
  char_type do_thousands_sep() const override { return ','; }
  string_type do_grouping() const override { return "\3"; }
};

struct stats {
  struct entry {
    bool operator<(entry const& o) const { return value_ < o.value_; }
    std::uint64_t msg_id_, value_;
  };

  stats() = default;
  stats(std::string name, std::uint64_t count_so_far)
      : name_{std::move(name)}, values_{count_so_far} {}

  void add(uint64_t msg_id, std::uint64_t value) {
    values_.emplace_back(entry{msg_id, value});
    sum_ += value;
  }

  std::string name_;
  std::vector<entry> values_;
  std::uint64_t sum_{};
};

struct category {
  category() = default;
  explicit category(std::string name) : name_(std::move(name)) {}

  std::string name_;
  std::map<std::string, stats> stats_;
};

stats::entry quantile(std::vector<stats::entry> const& sorted_values,
                      double q) {
  if (q == 1.0) {
    return sorted_values.back();
  } else {
    return sorted_values[std::min(
        static_cast<std::size_t>(std::round(q * (sorted_values.size() - 1))),
        sorted_values.size() - 1)];
  }
}

void print_category(category& cat,
                    std::uint64_t count,
                    bool const compact,
                    int const top) {
  std::cout << "\n"
            << cat.name_ << "\n"
            << std::string(cat.name_.size(), '=') << "\n"
            << std::endl;
  for (auto& s : cat.stats_) {
    auto& stat = s.second;
    if (stat.values_.empty()) {
      continue;
    }
    utl::sort(stat.values_);
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

      auto const from = static_cast<std::uint64_t>(
          std::max(static_cast<std::int64_t>(0L),
                   static_cast<std::int64_t>(stat.values_.size()) -
                       static_cast<std::int64_t>(top)));
      for (auto i = from; i != stat.values_.size(); ++i) {
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
          << stat.name_ << "\n      average: " << std::right << std::setw(15)
          << std::setprecision(2) << std::fixed << avg
          << "\n          max: " << std::right << std::setw(12)
          << std::max_element(begin(stat.values_), end(stat.values_))->value_
          << "\n  99 quantile: " << std::right << std::setw(12)
          << quantile(stat.values_, 0.99).value_
          << "\n  90 quantile: " << std::right << std::setw(12)
          << quantile(stat.values_, 0.9).value_
          << "\n  80 quantile: " << std::right << std::setw(12)
          << quantile(stat.values_, 0.8).value_
          << "\n  50 quantile: " << std::right << std::setw(12)
          << quantile(stat.values_, 0.5).value_
          << "\n          min: " << std::right << std::setw(12)
          << std::min_element(begin(stat.values_), end(stat.values_))->value_
          << "\n"
          << std::endl;
    }
  }
}

namespace motis {

int batch(int ac, char** av) {
  auto data_path = fs::path{"data"};
  auto queries_path = fs::path{"queries.txt"};
  auto responses_path = fs::path{"responses.txt"};
  auto mt = true;

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("multithreading,mt", po::value(&mt)->default_value(mt))  //
      ("queries,q", po::value(&queries_path)->default_value(queries_path),
       "queries file")  //
      ("responses,r", po::value(&responses_path)->default_value(responses_path),
       "response file");
  add_data_path_opt(desc, data_path);

  auto vm = parse_opt(ac, av, desc);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto queries = std::vector<std::string_view>{};
  auto f = cista::mmap{queries_path.generic_string().c_str(),
                       cista::mmap::protection::READ};
  utl::for_each_line(utl::cstr{f.view()},
                     [&](utl::cstr s) { queries.push_back(s.view()); });

  auto const c = config::read(data_path / "config.yml");
  utl::verify(c.timetable_.has_value(), "timetable required");

  auto d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");

  auto response_time = stats{"response_time", 0U};

  struct state {};

  auto out = std::ofstream{responses_path};
  auto m = motis_instance{net::default_exec{}, d, c, ""};
  auto const compute_response = [&](state&, std::size_t const id) {
    UTL_START_TIMING(request);
    auto response = std::string{};
    try {
      m.qr_(
          {boost::beast::http::verb::get,
           boost::beast::string_view{queries.at(id)}, 11},
          [&](net::web_server::http_res_t const& res) {
            std::visit(
                [&](auto&& r) {
                  using ResponseType = std::decay_t<decltype(r)>;
                  if constexpr (std::is_same_v<ResponseType,
                                               net::web_server::string_res_t>) {
                    response = r.body();
                    if (response.empty()) {
                      std::cout << "empty response for " << id << ": "
                                << queries.at(id) << " [status=" << r.result()
                                << "]\n";
                    }
                  } else {
                    throw utl::fail("not a valid response type: {}",
                                    cista::type_str<ResponseType>());
                  }
                },
                res);
          },
          false);
    } catch (std::exception const& e) {
      std::cerr << "ERROR IN QUERY " << id << ": " << e.what() << "\n";
    }
    return std::pair{UTL_GET_TIMING_MS(request), std::move(response)};
  };

  auto const pt = utl::activate_progress_tracker("batch");
  pt->in_high(queries.size());
  if (mt) {
    utl::parallel_ordered_collect_threadlocal<state>(
        queries.size(), compute_response,
        [&](std::size_t const id,
            std::pair<std::uint64_t, std::string> const& s) {
          response_time.add(id, s.first);
          out << s.second << "\n";
        },
        pt->update_fn());
  } else {
    auto s = state{};
    for (auto i = 0U; i != queries.size(); ++i) {
      compute_response(s, i);
      pt->increment();
    }
  }

  auto cat = category{};
  cat.name_ = "response_time";
  cat.stats_.emplace("response_time", std::move(response_time));
  std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));
  print_category(cat, queries.size(), false, 10U);

  return 0U;
}

}  // namespace motis
