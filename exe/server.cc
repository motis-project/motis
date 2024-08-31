#include "fmt/core.h"

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/read_file.h"

#include "net/run.h"

#include "osr/lookup.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"

#include "icc/data.h"
#include "icc/elevators/elevators.h"
#include "icc/endpoints/elevators.h"
#include "icc/endpoints/footpaths.h"
#include "icc/endpoints/graph.h"
#include "icc/endpoints/levels.h"
#include "icc/endpoints/matches.h"
#include "icc/endpoints/osr_routing.h"
#include "icc/endpoints/platforms.h"
#include "icc/endpoints/routing.h"
#include "icc/endpoints/update_elevator.h"
#include "icc/match_platforms.h"
#include "icc/point_rtree.h"
#include "icc/tt_location_rtree.h"
#include "icc/update_rtt_td_footpaths.h"

namespace asio = boost::asio;
namespace http = boost::beast::http;
namespace n = nigiri;
namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace json = boost::json;

using namespace icc;

template <typename T, typename... Args>
void GET(net::query_router& r, std::string target, Args&&... args) {
  if ((is_not_null(args) && ...)) {
    r.get(std::move(target), T{deref(args)...});
  }
}

template <typename T, typename... Args>
void POST(net::query_router& r, std::string target, Args&&... args) {
  if ((is_not_null(args) && ...)) {
    r.post(std::move(target), T{deref(args)...});
  }
}

int main(int ac, char** av) {
  if (ac != 2U) {
    return 1;
  }

  auto d = data{av[1]};
  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr = net::query_router{};
  //                .post("/api/matches", ep::matches{d})
  //                .post("/api/elevators", ep::elevators{d})
  //                .post("/api/route", ep::osr_routing{d})
  //                .post("/api/levels", ep::levels{d})
  //                .post("/api/platforms", ep::platforms{d})
  //                .post("/api/graph", ep::graph{d})
  //                .post("/api/footpaths", ep::footpaths{d})
  //                .post("/api/update_elevator", ep::update_elevator{d})
  //                .get("/api/v1/plan", ep::routing{d});

  POST<ep::matches>(qr, "/api/matches", d);

  qr.serve_files("ui/build");
  qr.enable_cors();
  s.on_http_request(std::move(qr));

  auto ec = boost::system::error_code{};
  s.init("0.0.0.0", "8000", ec);
  s.run();
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  std::cout << "listening on 0.0.0.0:8000\n";
  net::run(ioc)();
}