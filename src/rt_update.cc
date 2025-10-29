#include "motis/rt_update.h"

#include <memory>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/constants.h"
#include "motis/data.h"
#include "motis/elevators/update_elevators.h"
#include "motis/http_client.h"
#include "motis/railviz.h"
#include "motis/rt/auser.h"
#include "motis/rt/rt_metrics.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

asio::awaitable<ptr<elevators>> update_elevators(config const& c,
                                                 data const& d,
                                                 http_client& client,
                                                 n::rt_timetable& new_rtt) {
  utl::verify(c.has_elevators() && c.get_elevators()->url_ && c.timetable_,
              "elevator update requires settings for timetable + elevators");
  auto const res =
      co_await client.get(boost::urls::url{*c.get_elevators()->url_},
                          c.get_elevators()->headers_.value_or(headers_t{}));
  co_return update_elevators(c, d, get_http_body(res), new_rtt);
}

struct gtfs_rt_endpoint {
  config::timetable::dataset::rt ep_;
  n::source_idx_t src_;
  std::string tag_;
  gtfsrt_metrics metrics_;
};

struct auser_endpoint {
  config::timetable::dataset::rt ep_;
  n::source_idx_t src_;
  std::string tag_;
  vdvaus_metrics metrics_;
};

struct request_wait_logger {
  template <typename Executor>
  request_wait_logger(Executor const& executor, std::string url)
      : timer_{executor}, url_{std::move(url)} {}

  void cancel() { timer_.cancel(); }

  asio::steady_timer timer_;
  std::string url_;
};

void run_rt_update(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto client = http_client{};
        client.timeout_ = std::chrono::seconds{c.timetable_->http_timeout_};

        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        auto start_wait_logger = [&](std::string url) {
          auto ctx =
              std::make_shared<request_wait_logger>(executor, std::move(url));
          asio::co_spawn(
              executor,
              [ctx]() -> awaitable<void> {
                while (true) {
                  ctx->timer_.expires_after(std::chrono::seconds{5});
                  auto timer_ec = boost::system::error_code{};
                  co_await ctx->timer_.async_wait(
                      asio::redirect_error(asio::use_awaitable, timer_ec));
                  if (timer_ec == asio::error::operation_aborted) {
                    co_return;
                  }

                  fmt::println("{} waiting for ...", ctx->url_);
                }
              },
              asio::detached);
          return ctx;
        };

        auto stop_wait_logger = [&](std::shared_ptr<request_wait_logger>& ctx) {
          if (ctx) {
            ctx->cancel();
            ctx.reset();
          }
        };

        auto const whitelist = hash_set<std::string_view>{
            R"(https://gtfs.bus-tracker.fr/gtfs-rt/yelo)",
            R"(https://gtfs.bus-tracker.fr/gtfs-rt/tcar/trip-updates)",
            R"(https://gtfs.bus-tracker.fr/gtfs-rt/tcl)",
            R"(https://gtfs.bus-tracker.fr/gtfs-rt/tango/trip-updates)",
            R"(https://gtfs.bus-tracker.fr/gtfs-rt/lia/trip-updates)",
            R"(http://rt.gtfs.baguette.pirnet.si/gtfs-rt/HZPP/trip_updates.pb)",
            R"(http://rt.gtfs.baguette.pirnet.si/gtfs-rt/apSisak/trip_updates.pb)",
            R"(http://rt.gtfs.baguette.pirnet.si/gtfs-rt/prometSplit/trip_updates.pb)",
            R"(http://rt.gtfs.baguette.pirnet.si/gtfs-rt/Srbijavoz_ghl/trip_updates.pb)",
            R"(https://rt.gtfs.derp.si/sources/celje/trip_updates)",
            R"(https://rt.gtfs.derp.si/sources/lpp/all)",
            R"(https://rt.gtfs.derp.si/sources/marprom/trip_updates)",
            R"(https://rt.gtfs.derp.si/sources/ijpp/trip_updates)",
        };
        auto const endpoints = [&]() {
          auto endpoints =
              std::vector<std::variant<gtfs_rt_endpoint, auser_endpoint>>{};
          auto const metric_families =
              rt_metric_families{d.metrics_->registry_};
          for (auto const& [tag, dataset] : c.timetable_->datasets_) {
            if (dataset.rt_.has_value()) {
              auto const src = d.tags_->get_src(tag);
              for (auto const& ep : *dataset.rt_) {
                if (!whitelist.contains(ep.url_)) {
                  continue;
                }
                switch (ep.protocol_) {
                  case config::timetable::dataset::rt::protocol::gtfsrt:
                    endpoints.push_back(gtfs_rt_endpoint{
                        ep, src, tag, gtfsrt_metrics{tag, metric_families}});
                    break;
                  case config::timetable::dataset::rt::protocol::siri_json:
                  case config::timetable::dataset::rt::protocol::siri:
                    [[fallthrough]];
                  case config::timetable::dataset::rt::protocol::auser:
                    endpoints.push_back(auser_endpoint{
                        ep, src, tag, vdvaus_metrics{tag, metric_families}});
                    break;
                }
              }
            }
          }
          return endpoints;
        }();

        while (true) {
          // Remember when we started, so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          {
            auto t = utl::scoped_timer{"rt update"};

            // Create new real-time timetable.
            auto const today = std::chrono::time_point_cast<date::days>(
                std::chrono::system_clock::now());
            auto rtt = std::make_unique<n::rt_timetable>(
                c.timetable_->incremental_rt_update_
                    ? n::rt_timetable{*d.rt_->rtt_}
                    : n::rt::create_rt_timetable(*d.tt_, today));

            // Schedule updates for each real-time endpoint.
            if (!endpoints.empty()) {
              auto awaitables = utl::to_vec(
                  endpoints,
                  [&](std::variant<gtfs_rt_endpoint, auser_endpoint> const& x) {
                    return boost::asio::co_spawn(
                        executor,
                        [&]() -> awaitable<
                                  std::variant<n::rt::statistics,
                                               n::rt::vdv_aus::statistics>> {
                          auto ret = std::variant<n::rt::statistics,
                                                  n::rt::vdv_aus::statistics>{};
                          co_await std::visit(
                              utl::overloaded{
                                  [&](gtfs_rt_endpoint const& g)
                                      -> awaitable<void> {
                                    g.metrics_.updates_requested_.Increment();
                                    auto wait_logger =
                                        start_wait_logger(g.ep_.url_);
                                    try {
                                      auto const res = co_await client.get(
                                          boost::urls::url{g.ep_.url_},
                                          g.ep_.headers_.value_or(headers_t{}));
                                      stop_wait_logger(wait_logger);
                                      ret = n::rt::gtfsrt_update_buf(
                                          *d.tt_, *rtt, g.src_, g.tag_,
                                          get_http_body(res));
                                    } catch (std::exception const& e) {
                                      stop_wait_logger(wait_logger);
                                      g.metrics_.updates_error_.Increment();
                                      n::log(n::log_lvl::error, "motis.rt",
                                             "RT FETCH ERROR: tag={}, error={}",
                                             g.tag_, e.what());
                                      ret = n::rt::statistics{
                                          .parser_error_ = true,
                                          .no_header_ = true};
                                    }
                                  },
                                  [&](auser_endpoint const& a)
                                      -> awaitable<void> {
                                    a.metrics_.updates_requested_.Increment();
                                    auto& auser = d.auser_->at(a.ep_.url_);
                                    std::shared_ptr<request_wait_logger>
                                        wait_logger;
                                    try {
                                      auto const fetch_url_string =
                                          auser.fetch_url(a.ep_.url_);
                                      wait_logger =
                                          start_wait_logger(fetch_url_string);
                                      auto const fetch_url =
                                          boost::urls::url{fetch_url_string};
                                      fmt::println("[auser] fetch url: {}",
                                                   fetch_url.c_str());
                                      auto const res = co_await client.get(
                                          fetch_url,
                                          a.ep_.headers_.value_or(headers_t{}));
                                      stop_wait_logger(wait_logger);
                                      ret = auser.consume_update(
                                          get_http_body(res), *rtt);
                                    } catch (std::exception const& e) {
                                      stop_wait_logger(wait_logger);
                                      a.metrics_.updates_error_.Increment();
                                      n::log(n::log_lvl::error, "motis.rt",
                                             "VDV AUS FETCH ERROR: tag={}, "
                                             "url={}, error={}",
                                             a.tag_, a.ep_.url_, e.what());
                                      ret = nigiri::rt::vdv_aus::statistics{
                                          .error_ = true};
                                    }
                                  }},
                              x);
                          co_return ret;
                        },
                        asio::deferred);
                  });

              // Wait for all updates to finish
              auto [_, exceptions, stats] =
                  co_await asio::experimental::make_parallel_group(awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);

              //  Print statistics.
              for (auto const [ep, ex, s] :
                   utl::zip(endpoints, exceptions, stats)) {
                std::visit(
                    utl::overloaded{
                        [&](gtfs_rt_endpoint const& g) {
                          try {
                            if (ex) {
                              std::rethrow_exception(ex);
                            }

                            g.metrics_.updates_successful_.Increment();
                            g.metrics_.last_update_timestamp_
                                .SetToCurrentTime();
                            g.metrics_.update(std::get<n::rt::statistics>(s));

                            n::log(
                                n::log_lvl::info, "motis.rt",
                                "GTFS-RT update stats for tag={}, url={}: {}",
                                g.tag_, g.ep_.url_,
                                fmt::streamed(std::get<n::rt::statistics>(s)));
                          } catch (std::exception const& e) {
                            g.metrics_.updates_error_.Increment();
                            n::log(n::log_lvl::error, "motis.rt",
                                   "GTFS-RT update failed: tag={}, url={}, "
                                   "error={}",
                                   g.tag_, g.ep_.url_, e.what());
                          }
                        },
                        [&](auser_endpoint const& a) {
                          try {
                            if (ex) {
                              std::rethrow_exception(ex);
                            }

                            a.metrics_.updates_successful_.Increment();
                            a.metrics_.last_update_timestamp_
                                .SetToCurrentTime();
                            a.metrics_.update(
                                std::get<n::rt::vdv_aus::statistics>(s));

                            n::log(
                                n::log_lvl::info, "motis.rt",
                                "VDV AUS update stats for tag={}, url={}:\n{}",
                                a.tag_, a.ep_.url_,
                                fmt::streamed(
                                    std::get<n::rt::vdv_aus::statistics>(s)));
                          } catch (std::exception const& e) {
                            a.metrics_.updates_error_.Increment();
                            n::log(n::log_lvl::error, "motis.rt",
                                   "VDV AUS update failed: tag={}, url={}, "
                                   "error={}",
                                   a.tag_, a.ep_.url_, e.what());
                          }
                        }},
                    ep);
              }
            }

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators = c.has_elevators() && c.get_elevators()->url_
                                 ? co_await update_elevators(c, d, client, *rtt)
                                 : std::move(d.rt_->e_);
            d.rt_ = std::make_shared<rt>(std::move(rtt), std::move(elevators),
                                         std::move(railviz_rt));
          }

          // Schedule next update.
          timer.expires_at(
              start + std::chrono::seconds{c.timetable_->update_interval_});
          co_await timer.async_wait(
              asio::redirect_error(asio::use_awaitable, ec));
          if (ec == asio::error::operation_aborted) {
            co_return;
          }
        }
      },
      boost::asio::detached);
}

}  // namespace motis
