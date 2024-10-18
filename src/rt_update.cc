#include "motis/rt_update.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
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

awaitable<void> rt_update(config const& c,
                          nigiri::timetable const& tt,
                          tag_lookup const& tags,
                          std::shared_ptr<rt>& r,
                          std::chrono::seconds const http_timeout) {
  auto const t = utl::scoped_timer{"rt_update"};

  auto const no_hdr = headers_t{};
  auto gtfs_rt = std::vector<std::tuple<n::source_idx_t, boost::urls::url,
                                        awaitable<http_response>>>{};
  for (auto const& [tag, d] : c.timetable_->datasets_) {
    if (!d.rt_.has_value()) {
      continue;
    }

    auto const src = tags.get_src(tag);
    for (auto const& ep : *d.rt_) {
      auto const url = boost::urls::url{ep.url_};
      gtfs_rt.emplace_back(
          src, url,
          http_GET(url, ep.headers_.has_value() ? *ep.headers_ : no_hdr,
                   http_timeout));
    }
  }

  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  auto rtt = std::make_unique<n::rt_timetable>(
      c.timetable_->incremental_rt_update_
          ? n::rt_timetable{*r->rtt_}
          : n::rt::create_rt_timetable(tt, today));

  auto statistics = std::vector<n::rt::statistics>{};
  for (auto& [src, url, response] : gtfs_rt) {
    auto stats = n::rt::statistics{};
    auto const tag = tags.get_tag(src);
    try {
      auto const res = co_await std::move(response);
      stats = n::rt::gtfsrt_update_buf(
          tt, *rtt, src, tag,
          boost::beast::buffers_to_string(res.body().data()));
    } catch (std::exception const& e) {
      n::log(n::log_lvl::error, "motis.rt", "RT FETCH ERROR: tag={}, error={}",
             tag, e.what());
    }
    statistics.emplace_back(stats);
  }

  for (auto const [endpoint, stats] : utl::zip(gtfs_rt, statistics)) {
    auto const& [src, url, response] = endpoint;
    n::log(n::log_lvl::info, "motis.rt", "rt update stats for {}: {}",
           fmt::streamed(url), fmt::streamed(stats));
  }

  auto railviz_rt = std::make_unique<railviz_rt_index>(tt, *rtt);
  r = std::make_shared<rt>(std::move(rtt), std::move(r->e_),
                           std::move(railviz_rt));

  co_return;
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
          auto const start = std::chrono::steady_clock::now();

          try {
            co_await rt_update(
                c, tt, tags, r,
                std::chrono::seconds{c.timetable_->http_timeout_});
          } catch (std::exception const& e) {
            n::log(n::log_lvl::error, "motis.rt",
                   "EXCEPTION CAUGHT IN CRON: {}", e.what());
          } catch (...) {
            n::log(n::log_lvl::error, "motis.rt", "EXCEPTION CAUGHT IN CRON");
          }

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