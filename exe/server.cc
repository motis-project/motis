#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/asio/signal_set.hpp"
#include "boost/beast/core/string.hpp"
#include "boost/program_options.hpp"

#include "App.h"

#include "utl/init_from.h"

// #include "ui_files_res.h"

#include "motis/config.h"
#include "motis/cron.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/elevators.h"
#include "motis/endpoints/footpaths.h"
#include "motis/endpoints/graph.h"
#include "motis/endpoints/initial.h"
#include "motis/endpoints/levels.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/map/trips.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/one_to_many.h"
#include "motis/endpoints/osr_routing.h"
#include "motis/endpoints/platforms.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/tiles.h"
#include "motis/endpoints/trip.h"
#include "motis/endpoints/update_elevator.h"
#include "motis/http_server.h"
#include "motis/rt_update.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace asio = boost::asio;

namespace motis {

auto run_ioc(auto& ioc) {
  return [&]() {
    while (true) {
      try {
        ioc.run();
        break;
      } catch (std::exception const&) {
        continue;
      }
    }
  };
}

template <typename T>
void GET(auto& app, std::string target, auto&& from) {
  if (auto const x = utl::init_from<T>(from); x.has_value()) {
    handle_get(app, target, std::move(*x));
  }
}

int server(data d, config const& c) {
  auto const init = [&]() {
    auto app = uWS::App{};
    GET<ep::initial>(app, "/api/v1/map/initial", d);
    GET<ep::footpaths>(app, "/api/debug/footpaths", d);
    GET<ep::levels>(app, "/api/v1/map/levels", d);
    GET<ep::reverse_geocode>(app, "/api/v1/reverse-geocode", d);
    GET<ep::geocode>(app, "/api/v1/geocode", d);
    GET<ep::routing>(app, "/api/v1/plan", d);
    GET<ep::stop_times>(app, "/api/v1/stoptimes", d);
    GET<ep::trip>(app, "/api/v1/trip", d);
    GET<ep::trips>(app, "/api/v1/map/trips", d);
    GET<ep::stops>(app, "/api/v1/map/stops", d);
    GET<ep::one_to_many>(app, "/api/v1/one-to-many", d);

    if (c.tiles_) {
      utl::verify(d.tiles_ != nullptr, "tiles data not loaded");
      handle_get_generic_response(app, "/tiles/:z/:x/:y.mvt",
                                  ep::tiles{*d.tiles_});
    }

    app.options("/*", [](auto* res, auto*) { send_response(res, {}); });
    app.get("/*", [](auto* res, auto* req) {
      std::cout << "not found: " << req->getFullUrl() << std::endl;
      send_response(res, {});
    });
    return app;
  };

  auto const server_config = c.server_.value_or(config::server{});

  auto rt_update_thread = std::unique_ptr<std::thread>{};
  auto rt_update_ioc = std::unique_ptr<asio::io_context>{};
  if (c.requires_rt_timetable_updates()) {
    rt_update_ioc = std::make_unique<asio::io_context>();
    cron(*rt_update_ioc, std::chrono::seconds{c.timetable_->update_interval_},
         [&]() {
           asio::co_spawn(*rt_update_ioc, rt_update(c, *d.tt_, *d.tags_, d.rt_),
                          asio::detached);
         });
    rt_update_thread = std::make_unique<std::thread>(run_ioc(*rt_update_ioc));
  }

  auto listen_socket = static_cast<us_listen_socket_t*>(nullptr);

  //  auto signals = boost::asio::signal_set{workers, SIGINT, SIGTERM};
  //  signals.async_wait([&](boost::system::error_code const&, int) {
  //    std::cout << "shutting down..." << std::endl;
  //
  //    if (listen_socket != nullptr) {
  //      us_listen_socket_close(0, listen_socket);
  //    }
  //    workers.stop();
  //    if (rt_update_ioc != nullptr) {
  //      rt_update_ioc->stop();
  //    }
  //  });

  app.listen(server_config.port_,
             [&](us_listen_socket_t* s) {
               if (s != nullptr) {
                 listen_socket = s;
                 fmt::println(
                     "listening on {}:{}\nlocal link: http://localhost:{}",
                     server_config.host_, server_config.port_,
                     server_config.port_);
               } else {
                 fmt::println("no listen socket - something went wrong");
               }
             })
      .run();

  std::cout << "shutdown\n";

  //  workers.stop();
  //  for (auto& t : threads) {
  //    t.join();
  //  }
  //  if (rt_update_thread != nullptr) {
  //    rt_update_thread->join();
  //  }

  return 0;
}

int server(fs::path const& data_path) {
  try {
    auto const c = config::read(data_path / "config.yml");
    return server(data{data_path, c}, c);
  } catch (std::exception const& e) {
    std::cerr << "unable to start server: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace motis