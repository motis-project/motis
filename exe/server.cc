#include "fmt/core.h"

#include "boost/asio/io_context.hpp"
#include "boost/fiber/all.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/init_from.h"

#include "net/run.h"
#include "net/stop_handler.h"

#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/elevators.h"
#include "motis/endpoints/footpaths.h"
#include "motis/endpoints/graph.h"
#include "motis/endpoints/levels.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/osr_routing.h"
#include "motis/endpoints/platforms.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/update_elevator.h"

namespace asio = boost::asio;
namespace bpo = boost::program_options;

using namespace motis;

struct boost_fiber_executor {
  auto operator()(auto&& fn, net::web_server::http_res_cb_t const& cb) {
    boost::fibers::fiber([f = std::forward<decltype(fn)>(fn)]() {
      f();
    }).detach();
  }
};

template <typename T, typename From>
void GET(auto&& r, std::string target, From& from) {
  if (auto const x = utl::init_from<T>(from); x.has_value()) {
    r.get(std::move(target), std::move(*x));
  }
}

template <typename T, typename From>
void POST(auto&& r, std::string target, From& from) {
  if (auto const x = utl::init_from<T>(from); x.has_value()) {
    r.post(std::move(target), std::move(*x));
  }
}

int main(int ac, char** av) {
  auto d = data{};
  data::load(ac == 2U ? av[1] : "data", d);

  auto ioc = asio::io_context{};
  auto workers = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr = net::query_router{net::asio_exec({ioc, workers})};

  POST<ep::matches>(qr, "/api/matches", d);
  POST<ep::elevators>(qr, "/api/elevators", d);
  POST<ep::osr_routing>(qr, "/api/route", d);
  POST<ep::platforms>(qr, "/api/platforms", d);
  POST<ep::graph>(qr, "/api/graph", d);
  POST<ep::update_elevator>(qr, "/api/update_elevator", d);
  GET<ep::footpaths>(qr, "/api/debug/footpaths", d);
  GET<ep::levels>(qr, "/api/v1/levels", d);
  GET<ep::reverse_geocode>(qr, "/api/v1/reverse-geocode", d);
  GET<ep::geocode>(qr, "/api/v1/geocode", d);
  GET<ep::routing>(qr, "/api/v1/plan", d);

  qr.serve_files("ui/build");
  qr.enable_cors();
  s.on_http_request(std::move(qr));

  auto ec = boost::system::error_code{};
  s.init("0.0.0.0", "7999", ec);
  s.run();
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  auto work_guard = asio::make_work_guard(workers);
  auto threads = std::vector<std::thread>(
      static_cast<unsigned>(std::max(1U, std::thread::hardware_concurrency())));
  for (auto& t : threads) {
    t = std::thread(net::run(workers));
  }

  auto const stop = net::stop_handler(ioc, [&]() { s.stop(); });

  std::cout << "listening on 0.0.0.0:7999\n";
  net::run(ioc)();

  workers.stop();
  for (auto& t : threads) {
    t.join();
  }
}