#include "fmt/core.h"

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/init_from.h"

#include "net/run.h"

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

namespace asio = boost::asio;
namespace bpo = boost::program_options;

using namespace icc;

template <typename T, typename From>
void GET(net::query_router& r, std::string target, From& from) {
  if (auto const x = utl::init_from<T>(from); x.has_value()) {
    r.get(std::move(target), std::move(*x));
  }
}

template <typename T, typename From>
void POST(net::query_router& r, std::string target, From& from) {
  if (auto const x = utl::init_from<T>(from); x.has_value()) {
    r.post(std::move(target), std::move(*x));
  }
}

int main(int ac, char** av) {
  if (ac != 2U) {
    return 1;
  }

  auto d = data{};
  data::load(av[1], d);

  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr = net::query_router{};

  POST<ep::matches>(qr, "/api/matches", d);
  POST<ep::elevators>(qr, "/api/elevators", d);
  POST<ep::osr_routing>(qr, "/api/route", d);
  POST<ep::levels>(qr, "/api/levels", d);
  POST<ep::platforms>(qr, "/api/platforms", d);
  POST<ep::graph>(qr, "/api/graph", d);
  POST<ep::footpaths>(qr, "/api/footpaths", d);
  POST<ep::update_elevator>(qr, "/api/update_elevator", d);
  GET<ep::routing>(qr, "/api/v1/plan", d);

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