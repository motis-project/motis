#include "motis/rt_update.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#include "rfl/visit.hpp"

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
#include "motis/tag_lookup.h"
#include "motis/vdv_rt/connection.h"
#include "motis/vdv_rt/xml.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

asio::awaitable<ptr<elevators>> update_elevators(config const& c,
                                                 data const& d,
                                                 n::rt_timetable& new_rtt) {
  utl::verify(c.elevators_ && c.elevators_->url_ && c.timetable_,
              "elevator update requires settings for timetable + elevators");
  auto const res =
      co_await http_GET(boost::urls::url{*c.elevators_->url_},
                        c.elevators_->headers_.value_or(headers_t{}),
                        std::chrono::seconds{c.elevators_->http_timeout_});
  co_return update_elevators(c, d, get_http_body(res), new_rtt);
}

void run_rt_update(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto msg = transit_realtime::FeedMessage{};
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};
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
            auto gtfs_endpoints =
                std::vector<std::tuple<config::timetable::dataset::gtfs_rt,
                                       n::source_idx_t, std::string>>{};
            for (auto const& [tag, dataset] : c.timetable_->datasets_) {
              if (dataset.rt_.has_value()) {
                auto const src = d.tags_->get_src(tag);
                for (auto const& ep : *dataset.rt_) {
                  rfl::visit(
                      utl::overloaded{
                          [&](config::timetable::dataset::gtfs_rt const&
                                  gtfs_ep) {
                            gtfs_endpoints.emplace_back(gtfs_ep, src, tag);
                          },
                          [&](config::timetable::dataset::vdv_rt const&) {}},
                      ep.variant());
                }
              }
            }

            if (!gtfs_endpoints.empty()) {
              auto awaitables = utl::to_vec(gtfs_endpoints, [&](auto&& x) {
                auto const& [ep, src, tag] = x;
                return boost::asio::co_spawn(
                    executor,
                    [ep, src, tag, timeout, &rtt, &msg,
                     &d]() -> awaitable<n::rt::statistics> {
                      try {
                        auto const res = co_await http_GET(
                            boost::urls::url{ep.url_},
                            ep.headers_.value_or(headers_t{}), timeout);
                        co_return n::rt::gtfsrt_update_buf(
                            *d.tt_, *rtt, src, tag, get_http_body(res), msg);
                      } catch (std::exception const& e) {
                        n::log(n::log_lvl::error, "motis.rt",
                               "GTFS_RT FETCH ERROR: tag={}, error={}", tag,
                               e.what());
                        co_return n::rt::statistics{.parser_error_ = true,
                                                    .no_header_ = true};
                      }
                    },
                    asio::deferred);
              });

              // Wait for all updates to finish
              auto [idx, exceptions, stats] =
                  co_await asio::experimental::make_parallel_group(awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);

              //  Print statistics.
              for (auto const [i, ex, s] : utl::zip(idx, exceptions, stats)) {
                auto const [ep, src, tag] = gtfs_endpoints[i];
                try {
                  if (ex) {
                    std::rethrow_exception(ex);
                  }
                  n::log(n::log_lvl::info, "motis.rt",
                         "gtfs_rt update stats for tag={}, url={}: {}", tag,
                         ep.url_, fmt::streamed(s));
                } catch (std::exception const& e) {
                  n::log(n::log_lvl::error, "motis.rt",
                         "gtfs_rt update failed: tag={}, url={}, error={}", tag,
                         ep.url_, e.what());
                }
              }
            }

            if (d.vdv_rt_ != nullptr && !d.vdv_rt_->empty()) {
              auto vdv_awaitables = utl::to_vec(*d.vdv_rt_, [&](auto&& con) {
                return boost::asio::co_spawn(
                    executor,
                    [&con, timeout, &rtt,
                     &d]() -> awaitable<n::rt::vdv::statistics> {
                      try {
                        auto const res = co_await http_POST(
                            boost::urls::url{con.fetch_data_addr_},
                            vdv_rt::kHeaders, con.make_fetch_req(), timeout);
                        con.upd_.update(*rtt,
                                        vdv_rt::parse(get_http_body(res)));
                      } catch (std::exception const& e) {
                        n::log(n::log_lvl::error, "motis.rt",
                               "VDV_RT FETCH ERROR: tag={}, error={}",
                               d.tags_->get_tag(con.upd_.get_src()), e.what());
                      }
                      co_return con.upd_.get_stats();
                    },
                    asio::deferred);
              });

              auto [idx, exceptions, stats] =
                  co_await asio::experimental::make_parallel_group(
                      vdv_awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);

              for (auto const [i, ex, s] : utl::zip(idx, exceptions, stats)) {
                auto const& con = (*d.vdv_rt_)[i];
                try {
                  if (ex) {
                    std::rethrow_exception(ex);
                  }
                  n::log(n::log_lvl::info, "motis.rt",
                         "vdv_rt update stats for tag={}, server_url={}: {}",
                         d.tags_->get_tag(con.upd_.get_src()),
                         con.cfg_.server_url_, fmt::streamed(s));

                } catch (std::exception const& e) {
                  n::log(n::log_lvl::error, "motis.rt",
                         "vdv_rt update failed: tag={}, url={}, error={}",
                         d.tags_->get_tag(con.upd_.get_src()),
                         con.cfg_.server_url_, e.what());
                }
              }
            }

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators = c.elevators_ && c.elevators_->url_
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
