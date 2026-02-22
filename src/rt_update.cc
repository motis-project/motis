#include "motis/rt_update.h"

#include <filesystem>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#include "utl/read_file.h"
#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/update_elevators.h"
#include "motis/http_req.h"
#include "motis/railviz.h"
#include "motis/rt/auser.h"
#include "motis/rt/rt_metrics.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
namespace fs = std::filesystem;
using asio::awaitable;

namespace motis {

asio::awaitable<ptr<elevators>> update_elevators(config const& c,
                                                 data const& d,
                                                 n::rt_timetable& new_rtt) {
  utl::verify(c.has_elevators() && c.get_elevators()->url_ && c.timetable_,
              "elevator update requires settings for timetable + elevators");
  auto const res =
      co_await http_GET(boost::urls::url{*c.get_elevators()->url_},
                        c.get_elevators()->headers_.value_or(headers_t{}),
                        std::chrono::seconds{c.get_elevators()->http_timeout_});
  co_return update_elevators(c, d, get_http_body(res), new_rtt);
}

std::string get_dump_path(auto&& ep) {
  auto const normalize = [](std::string const& x) {
    auto ret = std::string{};
    ret.resize(x.size());
    for (auto [to, from] : utl::zip(ret, x)) {
      auto const c = from;
      if (('0' <= c && c <= '9') ||  //
          ('a' <= c && c <= 'z') ||  //
          ('A' <= c && c <= 'Z')) {
        to = c;
      } else {
        to = '_';
      }
    }
    return ret;
  };
  return fmt::format("dump_rt/{}-{}", ep.tag_, normalize(ep.ep_.url_));
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

void run_rt_update(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto const dump_rt = fs::is_directory("dump_rt");
        if (dump_rt) {
          fmt::println("WARNING: DUMPING TO dump_rt\n");
        }

        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        auto const endpoints = [&]() {
          auto endpoints =
              std::vector<std::variant<gtfs_rt_endpoint, auser_endpoint>>{};
          auto const metric_families =
              rt_metric_families{d.metrics_->registry_};
          for (auto const& [tag, dataset] : c.timetable_->datasets_) {
            if (dataset.rt_.has_value()) {
              auto const src = d.tags_->get_src(tag);
              for (auto const& ep : *dataset.rt_) {
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
            auto const timeout =
                std::chrono::seconds{c.timetable_->http_timeout_};

            using stats_t =
                std::variant<n::rt::statistics, n::rt::vdv_aus::statistics>;
            if (c.timetable_->canned_rt_) {
              fmt::println("WARNING: READING CANNED RT");

              auto const stats =
                  utl::to_vec(endpoints, [&](auto&& ep) -> stats_t {
                    try {
                      return utl::visit(
                          ep,
                          [&](gtfs_rt_endpoint const& g) -> stats_t {
                            auto const path = get_dump_path(g);
                            auto const body = utl::read_file(path.c_str());
                            if (body.has_value()) {
                              return n::rt::gtfsrt_update_buf(
                                  *d.tt_, *rtt, g.src_, g.tag_, *body);
                            } else {
                              return n::rt::statistics{.parser_error_ = true};
                            }
                          },
                          [&](auser_endpoint const& a) -> stats_t {
                            auto const path = get_dump_path(a);
                            auto& auser = d.auser_->at(a.ep_.url_);
                            auto const body = utl::read_file(path.c_str());
                            if (body.has_value()) {
                              return auser.consume_update(*body, *rtt);
                            } else {
                              return n::rt::vdv_aus::statistics{.error_ = true};
                            }
                          });
                    } catch (std::exception const& e) {
                      std::cout << "EXCEPTION: " << e.what() << "\n";
                      return n::rt::statistics{.parser_error_ = true};
                    }
                  });

              for (auto const [s, ep] : utl::zip(stats, endpoints)) {
                utl::visit(
                    ep,
                    [&](gtfs_rt_endpoint const& g) {
                      n::log(n::log_lvl::info, "motis.rt",
                             "GTFS-RT update stats for tag={}, url={}: {}",
                             g.tag_, g.ep_.url_,
                             fmt::streamed(std::get<n::rt::statistics>(s)));
                    },
                    [&](auser_endpoint const& a) {
                      n::log(n::log_lvl::info, "motis.rt",
                             "VDV AUS update stats for tag={}, url={}:\n{}",
                             a.tag_, a.ep_.url_,
                             fmt::streamed(
                                 std::get<n::rt::vdv_aus::statistics>(s)));
                    });
              }
            } else if (!endpoints.empty()) {
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
                                    try {
                                      auto const res = co_await http_GET(
                                          boost::urls::url{g.ep_.url_},
                                          g.ep_.headers_.value_or(headers_t{}),
                                          timeout);
                                      auto const body = get_http_body(res);
                                      if (dump_rt) {
                                        std::ofstream{get_dump_path(g)}.write(
                                            body.c_str(),
                                            static_cast<long>(body.size()));
                                      }
                                      ret = n::rt::gtfsrt_update_buf(
                                          *d.tt_, *rtt, g.src_, g.tag_, body);
                                    } catch (std::exception const& e) {
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
                                    try {
                                      auto const fetch_url = boost::urls::url{
                                          auser.fetch_url(a.ep_.url_)};
                                      fmt::println("[auser] fetch url: {}",
                                                   fetch_url.c_str());
                                      auto const res = co_await http_GET(
                                          fetch_url,
                                          a.ep_.headers_.value_or(headers_t{}),
                                          timeout);
                                      auto body = get_http_body(res);
                                      if (dump_rt) {
                                        std::ofstream{get_dump_path(a)}.write(
                                            body.c_str(),
                                            static_cast<long>(body.size()));
                                      }
                                      ret = auser.consume_update(body, *rtt,
                                                                 true);
                                    } catch (std::exception const& e) {
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

            // Update lbs.
            rtt->update_lbs(*d.tt_);

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators = c.has_elevators() && c.get_elevators()->url_
                                 ? co_await update_elevators(c, d, *rtt)
                                 : std::move(d.rt_->e_);
            auto new_rt = std::make_shared<rt>(
                std::move(rtt), std::move(elevators), std::move(railviz_rt));
            std::atomic_store(&d.rt_, std::move(new_rt));
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
