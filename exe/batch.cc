#include "conf/configuration.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>

#include "boost/json/serialize.hpp"
#include "boost/json/value_from.hpp"

#include "utl/init_from.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/timing.h"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

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

  auto queries = std::vector<api::plan_params>{};
  {
    auto f = cista::mmap{queries_path.generic_string().c_str(),
                         cista::mmap::protection::READ};
    utl::for_each_token(utl::cstr{f.view()}, '\n', [&](utl::cstr s) {
      queries.push_back(api::plan_params{boost::urls::url{s.view()}.params()});
    });
  }

  auto const c = config::read(data_path / "config.yml");
  utl::verify(c.timetable_.has_value(), "timetable required");

  auto d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");

  auto mtx = std::mutex{};
  auto out = std::ofstream{responses_path};
  auto total = std::atomic_uint64_t{};
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const compute_response = [&](std::size_t const id) {
    UTL_START_TIMING(total);
    auto response = routing(queries.at(id).to_url("/api/v1/plan"));
    UTL_STOP_TIMING(total);

    auto const timing = static_cast<std::uint64_t>(UTL_TIMING_MS(total));
    response.debugOutput_.emplace("id", id);
    response.debugOutput_.emplace("timing", timing);
    {
      auto const lock = std::scoped_lock{mtx};
      out << json::serialize(json::value_from(response)) << "\n";
    }
    total += timing;
  };

  if (mt) {
    utl::parallel_for_run(queries.size(), compute_response);
  } else {
    for (auto i = 0U; i != queries.size(); ++i) {
      compute_response(i);
    }
  }

  std::cout << "AVG: "
            << (static_cast<double>(total) /
                static_cast<double>(queries.size()))
            << "\n";

  return 0U;
}

}  // namespace motis