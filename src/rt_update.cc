#include "motis/rt_update.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/http_req.h"
#include "motis/railviz.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

template <typename Executor>
awaitable<n::rt::statistics> update(Executor executor,
                                    nigiri::timetable const& tt,
                                    n::source_idx_t src,
                                    std::string_view tag,
                                    config::timetable::dataset::rt ep,
                                    std::chrono::seconds http_timeout,
                                    n::rt_timetable& rtt) {
  return boost::asio::co_spawn(
      executor,
      [=, &rtt, &tt]() -> awaitable<n::rt::statistics> {
        try {
          auto const res = co_await http_GET(boost::urls::url{ep.url_},
                                             ep.headers_.value_or(headers_t{}),
                                             http_timeout);
          co_return n::rt::gtfsrt_update_buf(
              tt, rtt, src, tag,
              boost::beast::buffers_to_string(res.body().data()));
        } catch (std::exception const& e) {
          n::log(n::log_lvl::error, "motis.rt",
                 "RT FETCH ERROR: tag={}, error={}", tag, e.what());
          co_return n::rt::statistics{.parser_error_ = true,
                                      .no_header_ = true};
        }
      },
      asio::use_awaitable);
}

void run_rt_update(boost::asio::io_context& ioc,
                   config const& c,
                   nigiri::timetable const& tt,
                   tag_lookup const& tags,
                   std::shared_ptr<rt>& r) {
  boost::asio::co_spawn(
      ioc,
      [&c, &tt, &tags, &r]() -> awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};
        while (true) {
          // Remember when we started so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          {
            auto t = utl::scoped_timer{"rt update"};

            // Create new real-time timetable.
            auto const today = std::chrono::time_point_cast<date::days>(
                std::chrono::system_clock::now());
            auto rtt = std::make_unique<n::rt_timetable>(
                c.timetable_->incremental_rt_update_
                    ? n::rt_timetable{*r->rtt_}
                    : n::rt::create_rt_timetable(tt, today));

            // Schedule updates for each real-time endpoint.
            auto awaitables =
                std::vector<std::tuple<awaitable<n::rt::statistics>,
                                       std::string_view, boost::urls::url>>{};
            for (auto const& [tag, d] : c.timetable_->datasets_) {
              if (!d.rt_.has_value()) {
                continue;
              }

              auto const src = tags.get_src(tag);
              for (auto const& ep : *d.rt_) {
                auto const url = boost::urls::url{ep.url_};
                awaitables.emplace_back(
                    update(executor, tt, src, tag, ep,
                           std::chrono::seconds{c.timetable_->http_timeout_},
                           *rtt),
                    tag, url);
              }
            }

            // Wait for all updates to finish and print statistics.
            for (auto& [stats, tag, url] : awaitables) {
              try {
                n::log(n::log_lvl::info, "motis.rt",
                       "rt update stats for tag={}, url={}: {}", tag,
                       fmt::streamed(url),
                       fmt::streamed(co_await std::move(stats)));
              } catch (std::exception const& e) {
                n::log(n::log_lvl::error, "motis.rt",
                       "rt update failed: tag={}, url={}, error={}", tag,
                       fmt::streamed(url), e.what());
              }
            }

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(tt, *rtt);
            r = std::make_shared<rt>(std::move(rtt), std::move(r->e_),
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