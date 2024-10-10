#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/run.h"
#include "net/stop_handler.h"
#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/cron.h"
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
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/tiles.h"
#include "motis/endpoints/trip.h"
#include "motis/endpoints/update_elevator.h"
#include "motis/rt_update.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace asio = boost::asio;

namespace motis {

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

int server(int ac, char** av) {
  auto data_path = fs::path{"data"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("data,d", bpo::value(&data_path)->default_value(data_path), "data path");

  auto const pos_desc = bpo::positional_options_description{}.add("data", -1);

  auto vm = bpo::variables_map{};
  bpo::store(
      bpo::command_line_parser(ac, av).options(desc).positional(pos_desc).run(),
      vm);
  bpo::notify(vm);

  auto c = config::read(data_path / "config.yml");
  auto d = data{std::move(data_path), c};

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
  GET<ep::stop_times>(qr, "/api/v1/stoptimes", d);
  GET<ep::trip>(qr, "/api/v1/trip", d);

  if (c.tiles_) {
    utl::verify(d.tiles_ != nullptr, "tiles data not loaded");
    qr.route("GET", "/tiles/.*", ep::tiles{*d.tiles_});
  }

  qr.serve_files("ui/build");
  qr.enable_cors();
  s.on_http_request(std::move(qr));

  auto const server_config = c.server_.value_or(config::server{});
  auto ec = boost::system::error_code{};
  s.init(server_config.host_, server_config.port_, ec);
  s.run();

  if (c.requires_rt_timetable_updates()) {
    cron(ioc, std::chrono::seconds{c.timetable_->update_interval_}, [&]() {
      boost::asio::co_spawn(workers, rt_update(c, *d.tt_, *d.tags_, d.rt_),
                            boost::asio::detached);
    });
  }

  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  auto const work_guard = asio::make_work_guard(workers);
  auto threads = std::vector<std::thread>(
      static_cast<unsigned>(std::max(1U, server_config.n_threads_)));
  for (auto& t : threads) {
    t = std::thread(net::run(workers));
  }

  auto const stop = net::stop_handler(ioc, [&]() {
    fmt::println("shutdown");
    s.stop();
    ioc.stop();
  });

  fmt::println("listening on {}:{}\nlocal link: http://localhost:{}",
               server_config.host_, server_config.port_, server_config.port_);
  net::run(ioc)();

  workers.stop();
  for (auto& t : threads) {
    t.join();
  }

  return 0;
}

}  // namespace motis