#if defined(_MSC_VER)
// needs to be the first to include WinSock.h
#include "boost/asio.hpp"
#endif

#include <fstream>
#include <iostream>

#include "conf/configuration.h"

#include "utl/init_from.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"

#include "ctx/ctx.h"

#include "net/web_server/web_server.h"

#include "motis/config.h"
#include "motis/ctx_data.h"
#include "motis/ctx_exec.h"
#include "motis/data.h"
#include "motis/gbfs/update.h"
#include "motis/motis_instance.h"
#include "motis/rt_update.h"

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
  auto n_threads = std::thread::hardware_concurrency();
  auto rt = false;

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("n_threads,nt", po::value(&n_threads)->default_value(n_threads))  //
      ("queries,q", po::value(&queries_path)->default_value(queries_path),
       "queries file")  //
      ("responses,r", po::value(&responses_path)->default_value(responses_path),
       "response file")  //
      ("rt", po::bool_switch(&rt),
       "apply a canned rt update (dump_rt/ in the working directory, written "
       "by a server run with an existing dump_rt directory) before running "
       "the queries");
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

  fmt::println("hardware_concurrency = {}, using {} threads",
               std::thread::hardware_concurrency(), n_threads);

  if (rt) {
    apply_canned_rt_update(c, d);
  }
  gbfs::apply_canned_gbfs_update(c, d);

  auto response_time = stats{"response_time", 0U};

  auto out = std::ofstream{responses_path};

  // meta_router (ODM) dispatches sub-queries via ctx_call, which requires a
  // live ctx fiber operation to suspend/resume on. net::default_exec runs
  // handlers synchronously without such a context, so it must not be used
  // here -- mirror server.cc's ctx::scheduler + ctx_exec setup instead.
  auto scheduler = ctx::scheduler<ctx_data>{};
  auto m = motis_instance{ctx_exec{scheduler.runner_.ios(), scheduler}, d, c, ""};

  auto responses = std::vector<std::string>(queries.size());
  auto starts =
      std::vector<std::chrono::steady_clock::time_point>(queries.size());

  auto const pt = utl::activate_progress_tracker("batch");
  pt->in_high(queries.size());
  auto const start_batch = std::chrono::steady_clock::now();

  // ctx_exec completion callbacks are only ever invoked from the thread that
  // calls scheduler.runner_.run() below (it alone drives the ctx scheduler's
  // io_context), so this counter needs no synchronization.
  //
  // Deliberately NOT using runner_.run(n_threads, /*quit_on_ios_exit=*/true):
  // that shutdown path has ctx::runner::run() call work_stack_.clear() on
  // this thread while worker threads may still be concurrently popping from
  // the same work_stack_ -- a real race in ctx::runner, unrelated to this
  // tool, that a ThreadSanitizer run reproduced as a SEGV in
  // ctx::stack_manager::dealloc. Stopping the scheduler explicitly once all
  // queries have completed (mirroring server.cc's shutdown handler) avoids
  // that path entirely: work_stack_.stop() lets each worker's poll() loop
  // exit on its own before being joined.
  auto completed = std::size_t{0U};
  auto const on_query_done = [&] {
    if (++completed == queries.size()) {
      scheduler.runner_.stop();
    }
  };

  for (auto id = std::size_t{0U}; id != queries.size(); ++id) {
    starts[id] = std::chrono::steady_clock::now();
    try {
      m.qr_(
          {boost::beast::http::verb::get,
           boost::beast::string_view{queries.at(id)}, 11},
          [&, id](net::web_server::http_res_t const& res) {
            std::visit(
                [&](auto&& r) {
                  using ResponseType = std::decay_t<decltype(r)>;
                  if constexpr (std::is_same_v<ResponseType,
                                               net::web_server::string_res_t>) {
                    responses[id] = r.body();
                    if (responses[id].empty()) {
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
            response_time.add(
                id, static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - starts[id])
                            .count()));
            pt->increment();
            on_query_done();
          },
          false);
    } catch (std::exception const& e) {
      std::cerr << "ERROR IN QUERY " << id << ": " << e.what() << "\n";
      on_query_done();
    }
  }

  // runs the ctx scheduler's io loop + worker threads on this thread until
  // on_query_done() above calls scheduler.runner_.stop().
  if (!queries.empty()) {
    scheduler.runner_.run(n_threads);
  }

  for (auto const& response : responses) {
    out << response << "\n";
  }

  fmt::println("Processed {} queries in {:%T}", queries.size(),
               std::chrono::steady_clock::now() - start_batch);

  auto cat = category{};
  cat.name_ = "response_time";
  cat.stats_.emplace("response_time", std::move(response_time));
  std::cout.imbue(std::locale(std::locale::classic(), new thousands_sep));
  print_category(cat, queries.size(), false, 10U);

  return 0U;
}

}  // namespace motis
