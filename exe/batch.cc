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
  utl::for_each_token(utl::cstr{f.view()}, '\n',
                      [&](utl::cstr s) { queries.push_back(s.view()); });

  auto const c = config::read(data_path / "config.yml");
  utl::verify(c.timetable_.has_value(), "timetable required");

  auto d = data{data_path, c};
  utl::verify(d.tt_, "timetable required");

  struct state {};

  auto out = std::ofstream{responses_path};
  auto total = std::atomic_uint64_t{};
  auto m = motis_instance{net::default_exec{}, d, c, ""};
  auto const compute_response = [&](state&, std::size_t const id) {
    UTL_START_TIMING(request);
    auto response = std::string{};
    try {
      m.qr_(
          net::web_server::http_req_t{boost::beast::http::verb::get,
                                      boost::beast::string_view{queries.at(id)},
                                      11},
          [&](net::web_server::http_res_t const& res) {
            std::visit(
                [&](auto&& r) {
                  using ResponseType = std::decay_t<decltype(r)>;
                  if constexpr (std::is_same_v<ResponseType,
                                               net::web_server::string_res_t>) {
                    response = r.body();
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
    total += static_cast<std::uint64_t>(UTL_GET_TIMING_MS(request));
    return response;
  };

  auto const pt = utl::activate_progress_tracker("batch");
  pt->in_high(queries.size());
  if (mt) {
    utl::parallel_ordered_collect_threadlocal<state>(
        queries.size(), compute_response,
        [&](std::size_t, std::string const& s) { out << s << "\n"; },
        pt->update_fn());
  } else {
    auto s = state{};
    for (auto i = 0U; i != queries.size(); ++i) {
      compute_response(s, i);
      pt->increment();
    }
  }

  std::cout << "AVG: "
            << (static_cast<double>(total) /
                static_cast<double>(queries.size()))
            << "\n";

  return 0U;
}

}  // namespace motis