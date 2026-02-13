#include <memory>
#include <thread>

#include "boost/asio/io_context.hpp"

#include "net/web_server/query_router.h"

#include "utl/set_thread_name.h"

#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/elevators.h"
#include "motis/endpoints/graph.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/initial.h"
#include "motis/endpoints/levels.h"
#include "motis/endpoints/map/flex_locations.h"
#include "motis/endpoints/map/rental.h"
#include "motis/endpoints/map/routes.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/map/trips.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/metrics.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/one_to_many.h"
#include "motis/endpoints/one_to_many_post.h"
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

namespace motis {

struct io_thread {
  template <typename Fn>
  io_thread(char const* name, Fn&& f) {
    ioc_ = std::make_unique<boost::asio::io_context>();
    t_ = std::make_unique<std::thread>(
        [ioc = ioc_.get(), name, f = std::move(f)]() {
          utl::set_current_thread_name(name);
          f(*ioc);
          ioc->run();
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
  std::unique_ptr<boost::asio::io_context> ioc_;
};

template <typename Executor>
struct motis_instance {
  motis_instance(Executor&& exec,
                 data& d,
                 config const& c,
                 std::string_view motis_version)
      : qr_{std::forward<Executor>(exec)} {
    qr_.add_header("Server", fmt::format("MOTIS {}", motis_version));
    if (c.server_.value_or(config::server{}).data_attribution_link_) {
      qr_.add_header("Link", fmt::format("<{}>; rel=\"license\"",
                                         *c.server_->data_attribution_link_));
    }

    POST<ep::matches>("/api/matches", d);
    POST<ep::elevators>("/api/elevators", d);
    POST<ep::osr_routing>("/api/route", d);
    POST<ep::platforms>("/api/platforms", d);
    POST<ep::graph>("/api/graph", d);
    GET<ep::transfers>("/api/debug/transfers", d);
    GET<ep::flex_locations>("/api/debug/flex", d);
    GET<ep::levels>("/api/v1/map/levels", d);
    GET<ep::initial>("/api/v1/map/initial", d);
    GET<ep::reverse_geocode>("/api/v1/reverse-geocode", d);
    GET<ep::geocode>("/api/v1/geocode", d);
    GET<ep::routing>("/api/v1/plan", d);
    GET<ep::routing>("/api/v2/plan", d);
    GET<ep::routing>("/api/v3/plan", d);
    GET<ep::routing>("/api/v4/plan", d);
    GET<ep::routing>("/api/v5/plan", d);
    GET<ep::stop_times>("/api/v1/stoptimes", d);
    GET<ep::stop_times>("/api/v4/stoptimes", d);
    GET<ep::stop_times>("/api/v5/stoptimes", d);
    GET<ep::trip>("/api/v1/trip", d);
    GET<ep::trip>("/api/v2/trip", d);
    GET<ep::trip>("/api/v4/trip", d);
    GET<ep::trip>("/api/v5/trip", d);
    GET<ep::trips>("/api/v1/map/trips", d);
    GET<ep::trips>("/api/v4/map/trips", d);
    GET<ep::trips>("/api/v5/map/trips", d);
    GET<ep::stops>("/api/v1/map/stops", d);
    GET<ep::routes>("/api/experimental/map/routes", d);
    GET<ep::rental>("/api/v1/map/rentals", d);
    GET<ep::rental>("/api/v1/rentals", d);
    GET<ep::one_to_all>("/api/experimental/one-to-all", d);
    GET<ep::one_to_all>("/api/v1/one-to-all", d);
    GET<ep::one_to_many>("/api/v1/one-to-many", d);
    POST<ep::one_to_many_post>("/api/v1/one-to-many", d);

    if (!c.requires_rt_timetable_updates()) {
      // Elevator updates are not compatible with RT-updates.
      POST<ep::update_elevator>("/api/update_elevator", d);
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

  template <typename T, typename From>
  void GET(std::string target, From& from) {
    if (auto const x = utl::init_from<T>(from); x.has_value()) {
      qr_.get(std::move(target), std::move(*x));
    }
  }

  template <typename T, typename From>
  void POST(std::string target, From& from) {
    if (auto const x = utl::init_from<T>(from); x.has_value()) {
      qr_.post(std::move(target), std::move(*x));
    }
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
  }

  void stop() {
    rt_.stop();
    gbfs_.stop();
  }

  void join() {
    rt_.join();
    gbfs_.join();
  }

  net::query_router<Executor> qr_{};
  io_thread rt_, gbfs_;
};

}  // namespace motis
