#include "boost/asio/io_context.hpp"

#include "fmt/format.h"

#include "../deps/net/include/net/lb.h"
#include "net/run.h"
#include "net/stop_handler.h"
#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/enumerate.h"
#include "utl/init_from.h"
#include "utl/logging.h"
#include "utl/set_thread_name.h"

#include "ctx/ctx.h"

#include "motis/config.h"
#include "motis/ctx_data.h"
#include "motis/ctx_exec.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/elevators.h"
#include "motis/endpoints/graph.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/initial.h"
#include "motis/endpoints/levels.h"
#include "motis/endpoints/map/flex_locations.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/map/trips.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/metrics.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/one_to_many.h"
#include "motis/endpoints/osr_routing.h"
#include "motis/endpoints/platforms.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/tiles.h"
#include "motis/endpoints/transfers.h"
#include "motis/endpoints/trip.h"
#include "motis/endpoints/update_elevator.h"
#include "motis/gbfs/update.h"
#include "motis/metrics_registry.h"
#include "motis/rt_update.h"

namespace fs = std::filesystem;
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

struct io_thread {
  template <typename Fn>
  io_thread(char const* name, Fn&& f) {
    ioc_ = std::make_unique<asio::io_context>();
    t_ = std::make_unique<std::thread>([&, f = std::move(f)]() {
      utl::set_current_thread_name(name);
      f(*ioc_);
      ioc_->run();
    });
  }

  io_thread() = default;

  void stop() {
    if (ioc_ == nullptr) {
      return;
    }
    ioc_->stop();
  }

  void join() {
    if (t_ == nullptr) {
      return;
    }
    t_->join();
  }

  std::unique_ptr<std::thread> t_;
  std::unique_ptr<asio::io_context> ioc_;
};

struct motis_instance {
  motis_instance(data& d, config const& c, std::string_view motis_version) {
    qr_.add_header("Server", fmt::format("MOTIS {}", motis_version));
    if (c.server_.value_or(config::server{}).data_attribution_link_) {
      qr_.add_header("Link", fmt::format("<{}>; rel=\"license\"",
                                         *c.server_->data_attribution_link_));
    }

    POST<ep::matches>(qr_, "/api/matches", d);
    POST<ep::elevators>(qr_, "/api/elevators", d);
    POST<ep::osr_routing>(qr_, "/api/route", d);
    POST<ep::platforms>(qr_, "/api/platforms", d);
    POST<ep::graph>(qr_, "/api/graph", d);
    GET<ep::transfers>(qr_, "/api/debug/transfers", d);
    GET<ep::flex_locations>(qr_, "/api/debug/flex", d);
    GET<ep::levels>(qr_, "/api/v1/map/levels", d);
    GET<ep::initial>(qr_, "/api/v1/map/initial", d);
    GET<ep::reverse_geocode>(qr_, "/api/v1/reverse-geocode", d);
    GET<ep::geocode>(qr_, "/api/v1/geocode", d);
    GET<ep::routing>(qr_, "/api/v1/plan", d);
    GET<ep::routing>(qr_, "/api/v2/plan", d);
    GET<ep::routing>(qr_, "/api/v3/plan", d);
    GET<ep::routing>(qr_, "/api/v4/plan", d);
    GET<ep::stop_times>(qr_, "/api/v1/stoptimes", d);
    GET<ep::stop_times>(qr_, "/api/v4/stoptimes", d);
    GET<ep::trip>(qr_, "/api/v1/trip", d);
    GET<ep::trip>(qr_, "/api/v2/trip", d);
    GET<ep::trip>(qr_, "/api/v4/trip", d);
    GET<ep::trips>(qr_, "/api/v1/map/trips", d);
    GET<ep::trips>(qr_, "/api/v4/map/trips", d);
    GET<ep::stops>(qr_, "/api/v1/map/stops", d);
    GET<ep::one_to_all>(qr_, "/api/experimental/one-to-all", d);
    GET<ep::one_to_all>(qr_, "/api/v1/one-to-all", d);
    GET<ep::one_to_many>(qr_, "/api/v1/one-to-many", d);

    if (!c.requires_rt_timetable_updates()) {
      // Elevator updates are not compatible with RT-updates.
      POST<ep::update_elevator>(qr_, "/api/update_elevator", d);
    }

    if (c.tiles_) {
      utl::verify(d.tiles_ != nullptr, "tiles data not loaded");
      qr_.route("GET", "/tiles/", ep::tiles{*d.tiles_});
    }

    qr_.route("GET", "/metrics",
              ep::metrics{d.tt_.get(), d.tags_.get(), d.rt_, d.metrics_.get()});
    qr_.route("GET", "/gtfsrt",
              ep::gtfsrt{c, d.tt_.get(), d.tags_.get(), d.rt_});
    qr_.serve_files(c.server_.value_or(config::server{}).web_folder_);
    qr_.enable_cors();
  }

  void run(data& d, config const& c) {
    if (d.w_ && d.l_ && c.has_gbfs_feeds()) {
      gbfs_ = io_thread{"motis gbfs update", [&](boost::asio::io_context& ioc) {
                          gbfs::run_gbfs_update(ioc, c, *d.w_, *d.l_, d.gbfs_);
                        }};
    }

    if (c.requires_rt_timetable_updates()) {
      rt_ = io_thread{"motis rt update", [&](boost::asio::io_context& ioc) {
                        run_rt_update(ioc, c, d);
                      }};
    }

    sched_.runner_.run(c.n_threads());
    rt_.join();
    gbfs_.join();
  }

  void stop() {
    sched_.runner_.stop();
    rt_.stop();
    gbfs_.stop();
  }

  ctx::scheduler<ctx_data> sched_;
  net::query_router<ctx_exec> qr_{ctx_exec{sched_.runner_.ios(), sched_}};
  io_thread rt_, gbfs_;
};

int server(data d, config const& c, std::string_view const motis_version) {
  auto m = motis_instance{d, c, motis_version};

  auto s = net::web_server{m.sched_.runner_.ios()};
  s.set_timeout(std::chrono::minutes{5});
  s.on_http_request(m.qr_);

  auto ec = boost::system::error_code{};
  auto const server_config = c.server_.value_or(config::server{});
  s.init(server_config.host_, server_config.port_, ec);
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  auto lbs = std::vector<net::lb>{};
  if (c.server_.value_or(config::server{}).lbs_) {
    lbs = utl::to_vec(*c.server_.value_or(config::server{}).lbs_,
                      [&](std::string const& url) {
                        return net::lb{m.sched_.runner_.ios(), url, m.qr_};
                      });
  }

  auto const stop = net::stop_handler(m.sched_.runner_.ios(), [&]() {
    utl::log_info("motis.server", "shutdown");
    for (auto& lb : lbs) {
      lb.stop();
    }
    s.stop();
    m.stop();
  });

  utl::log_info(
      "motis.server",
      "n_threads={}, listening on {}:{}\nlocal link: http://localhost:{}",
      c.n_threads(), server_config.host_, server_config.port_,
      server_config.port_);

  for (auto const& lb : lbs) {
    lb.run();
  }
  s.run();
  m.run(d, c);

  return 0;
}

}  // namespace motis
