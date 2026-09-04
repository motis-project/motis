#include <memory>
#include <thread>

#include "boost/asio/io_context.hpp"

#include "net/web_server/enable_cors.h"
#include "net/web_server/query_router.h"
#include "net/web_server/responses.h"

#include "utl/helpers/algorithm.h"
#include "utl/set_thread_name.h"

#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/elevators.h"
#include "motis/endpoints/graph.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/health.h"
#include "motis/endpoints/initial.h"
#include "motis/endpoints/levels.h"
#include "motis/endpoints/map/flex_locations.h"
#include "motis/endpoints/map/rental.h"
#include "motis/endpoints/map/route_details.h"
#include "motis/endpoints/map/routes.h"
#include "motis/endpoints/map/shapes_debug.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/map/trips.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/mcp.h"
#include "motis/endpoints/metrics.h"
#include "motis/endpoints/ojp.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/one_to_many.h"
#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/osr_routing.h"
#include "motis/endpoints/platforms.h"
#include "motis/endpoints/refresh_itinerary.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/tiles.h"
#include "motis/endpoints/transfers.h"
#include "motis/endpoints/trip.h"
#include "motis/endpoints/update_elevator.h"
#include "motis/gbfs/update.h"
#include "motis/health.h"
#include "motis/metrics_registry.h"
#include "motis/rt_update.h"

namespace motis {

template <typename T>
concept uses_rt = requires(T const& t) { t.rt_; };

template <typename T>
concept uses_gbfs = requires(T const& t) { t.gbfs_; };

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
      : qr_{std::forward<Executor>(exec)},
        config_{&c},
        metrics_{d.metrics_.get()} {
    qr_.add_header("Server", fmt::format("MOTIS {}", motis_version));
    d.init_initial(motis_version);
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
    GET<ep::health>("/api/v1/health", d);
    GET<ep::geocode>("/api/v1/geocode", d);
    GET<ep::routing>("/api/v1/plan", d);
    GET<ep::routing>("/api/v2/plan", d);
    GET<ep::routing>("/api/v3/plan", d);
    GET<ep::routing>("/api/v4/plan", d);
    GET<ep::routing>("/api/v5/plan", d);
    GET<ep::routing>("/api/v6/plan", d);
    GET<ep::stop_times>("/api/v1/stoptimes", d);
    GET<ep::stop_times>("/api/v4/stoptimes", d);
    GET<ep::stop_times>("/api/v5/stoptimes", d);
    GET<ep::stop_times>("/api/v6/stoptimes", d);
    GET<ep::stop>("/api/v6/stop", d);
    GET<ep::trip>("/api/v1/trip", d);
    GET<ep::trip>("/api/v2/trip", d);
    GET<ep::trip>("/api/v4/trip", d);
    GET<ep::trip>("/api/v5/trip", d);
    GET<ep::trip>("/api/v6/trip", d);
    GET<ep::trips>("/api/v1/map/trips", d);
    GET<ep::trips>("/api/v4/map/trips", d);
    GET<ep::trips>("/api/v5/map/trips", d);
    GET<ep::trips>("/api/v6/map/trips", d);
    GET<ep::stops>("/api/v1/map/stops", d);
    GET<ep::stops>("/api/v6/map/stops", d);
    GET<ep::route_details>("/api/experimental/map/route-details", d);
    GET<ep::routes>("/api/experimental/map/routes", d);
    GET<ep::rental>("/api/v1/map/rentals", d);
    GET<ep::rental>("/api/v1/rentals", d);
    GET<ep::one_to_all>("/api/experimental/one-to-all", d);
    GET<ep::one_to_all>("/api/v1/one-to-all", d);
    GET<ep::one_to_all>("/api/v6/one-to-all", d);
    GET<ep::one_to_many>("/api/v1/one-to-many", d);
    GET<ep::refresh_itinerary>("/api/v6/refresh-itinerary", d);
    GET<ep::one_to_many_intermodal>("/api/experimental/one-to-many-intermodal",
                                    d);
    POST<ep::one_to_many_intermodal_post>(
        "/api/experimental/one-to-many-intermodal", d);
    POST<ep::one_to_many_post>("/api/v1/one-to-many", d);
    POST<ep::refresh_itinerary_post>("/api/v6/refresh-itinerary", d);
    POST<ep::routing_post>("/api/v6/plan", d);

    if (!c.requires_rt_timetable_updates()) {
      // Elevator updates are not compatible with RT-updates.
      POST<ep::update_elevator>("/api/update_elevator", d);
    }

    if (c.shapes_debug_api_enabled()) {
      utl::verify(d.w_ != nullptr && d.l_ != nullptr && d.tt_ != nullptr &&
                      d.tags_ != nullptr,
                  "data for shapes debug api not loaded");
      qr_.route("GET", "/api/experimental/shapes-debug/",
                ep::shapes_debug{c, d.w_.get(), d.l_.get(), d.tt_.get(),
                                 d.tags_.get()});
    }

    if (c.tiles_) {
      utl::verify(d.tiles_ != nullptr, "tiles data not loaded");
      qr_.route("GET", "/tiles/", ep::tiles{*d.tiles_});
    }

    qr_.route("POST", "/ojp20",
              ep::ojp{
                  .routing_ep_ = utl::init_from<ep::routing>(d),
                  .geocoding_ep_ = utl::init_from<ep::geocode>(d),
                  .stops_ep_ = utl::init_from<ep::stops>(d),
                  .stop_times_ep_ = utl::init_from<ep::stop_times>(d),
                  .trip_ep_ = utl::init_from<ep::trip>(d),
              });

    auto mcp = ep::mcp{.routing_ep_ = utl::init_from<ep::routing>(d),
                       .geocoding_ep_ = utl::init_from<ep::geocode>(d),
                       .motis_version_ = std::string{motis_version}};
    qr_.route("GET", "/api/mcp", mcp);  // answered with 405 (no SSE stream)
    qr_.route("POST", "/api/mcp", std::move(mcp));

    qr_.route("GET", "/metrics",
              ep::metrics{d.tt_.get(), d.tags_.get(), d.rt_, d.metrics_.get()});
    qr_.route("GET", "/gtfsrt",
              ep::gtfsrt{c, d.tt_.get(), d.tags_.get(), d.rt_});
    qr_.serve_files(c.server_.value_or(config::server{}).web_folder_);
    qr_.enable_cors();
  }

  template <typename T, typename From>
  void GET(std::string target, From& from) {
    if (auto x = utl::init_from<T>(from); x.has_value()) {
      register_health_gate<T>(target);
      qr_.get(std::move(target), std::move(*x));
    }
  }

  template <typename T, typename From>
  void POST(std::string target, From& from) {
    if (auto x = utl::init_from<T>(from); x.has_value()) {
      register_health_gate<T>(target);
      qr_.post(std::move(target), std::move(*x));
    }
  }

  template <typename T>
  void register_health_gate(std::string const& target) {
    auto needs_rt = false;
    auto needs_gbfs = false;
    if constexpr (uses_rt<T>) {
      needs_rt = config_->requires_rt_timetable_updates();
    }
    if constexpr (uses_gbfs<T>) {
      needs_gbfs = config_->has_gbfs_feeds();
    }
    if (needs_rt || needs_gbfs) {
      health_gated_prefixes_.push_back(target);
    }
  }

  // 503s gated endpoints while never-healthy
  // (server.when_unhealthy_return_503).
  void dispatch(net::web_server::http_req_t req,
                net::web_server::http_res_cb_t cb,
                bool is_ssl) {
    if (!health_gated_prefixes_.empty() &&
        config_->server_.value_or(config::server{})
            .when_unhealthy_return_503_ &&
        req.method() != boost::beast::http::verb::options) {
      auto const path = boost::urls::url_view{req.target()}.path();
      auto const gated = utl::any_of(
          health_gated_prefixes_,
          [&](std::string const& p) { return path.starts_with(p); });
      if (gated && !is_healthy(*config_, *metrics_)) {
        auto rep = net::reply{net::string_response(
            req,
            R"({"error":"motis is starting up: waiting for the initial )"
            R"(realtime/GBFS update"})",
            boost::beast::http::status::service_unavailable)};
        net::enable_cors(rep);
        return cb(std::move(rep));
      }
    }
    qr_(std::move(req), std::move(cb), is_ssl);
  }

  void run(data& d, config const& c) {
    if (d.w_ && d.l_ && c.has_gbfs_feeds()) {
      gbfs_ = io_thread{"motis gbfs update", [&](boost::asio::io_context& ioc) {
                          gbfs::run_gbfs_update(ioc, c, *d.w_, *d.l_, d.gbfs_,
                                                d.metrics_.get());
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
  config const* config_{};
  metrics_registry const* metrics_{};
  std::vector<std::string> health_gated_prefixes_{};
  io_thread rt_, gbfs_;
};

}  // namespace motis
