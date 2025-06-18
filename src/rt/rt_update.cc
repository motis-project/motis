#include "motis/rt/rt_update.h"

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
#include "motis/http_req.h"
#include "motis/railviz.h"
#include "motis/rt/rt_metrics.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
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

struct gtfs_rt_endpoint {
  config::timetable::dataset::rt ep_;
  n::source_idx_t src_;
  std::string tag_;
  gtfsrt_metrics metrics_;
};

void run_rt_update(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto msg = transit_realtime::FeedMessage{};
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        auto const endpoints = [&]() {
          auto endpoints = std::vector<gtfs_rt_endpoint>{};
          auto const metric_families =
              rt_metric_families{d.metrics_->registry_};
          for (auto const& [tag, dataset] : c.timetable_->datasets_) {
            if (dataset.rt_.has_value()) {
              auto const src = d.tags_->get_src(tag);
              for (auto const& ep : *dataset.rt_) {
                endpoints.push_back(
                    {ep, src, tag, gtfsrt_metrics{tag, metric_families}});
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

            if (!endpoints.empty()) {
              auto awaitables =
                  utl::to_vec(endpoints, [&](gtfs_rt_endpoint const& x) {
                    x.metrics_.updates_requested_.Increment();
                    return boost::asio::co_spawn(
                        executor,
                        [&]() -> awaitable<n::rt::statistics> {
                          try {
                            auto const res = co_await http_GET(
                                boost::urls::url{x.ep_.url_},
                                x.ep_.headers_.value_or(headers_t{}), timeout);
                            co_return n::rt::gtfsrt_update_buf(
                                *d.tt_, *rtt, x.src_, x.tag_,
                                get_http_body(res), msg);
                          } catch (std::exception const& e) {
                            n::log(n::log_lvl::error, "motis.rt",
                                   "RT FETCH ERROR: tag={}, error={}", x.tag_,
                                   e.what());
                            co_return n::rt::statistics{.parser_error_ = true,
                                                        .no_header_ = true};
                          }
                        },
                        asio::deferred);
                  });

              // Wait for all updates to finish
              auto [_, exceptions, stats] =
                  co_await asio::experimental::make_parallel_group(awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);

              //  Print statistics.
              for (auto const [endpoint, ex, s] :
                   utl::zip(endpoints, exceptions, stats)) {
                auto const& [ep, src, tag, metrics] = endpoint;
                try {
                  if (ex) {
                    std::rethrow_exception(ex);
                  }

                  metrics.updates_successful_.Increment();
                  metrics.last_update_timestamp_.SetToCurrentTime();
                  metrics.update(s);

                  n::log(n::log_lvl::info, "motis.rt",
                         "rt update stats for tag={}, url={}: {}", tag, ep.url_,
                         fmt::streamed(s));
                } catch (std::exception const& e) {
                  metrics.updates_error_.Increment();
                  n::log(n::log_lvl::error, "motis.rt",
                         "rt update failed: tag={}, url={}, error={}", tag,
                         ep.url_, e.what());
                }
              }
            }

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators = c.has_elevators() && c.get_elevators()->url_
                                 ? co_await update_elevators(c, d, *rtt)
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
