#include "motis/static_reload.h"

#include <algorithm>
#include <fstream>
#include <system_error>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/error.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/asio/this_coro.hpp"
#include "boost/asio/use_awaitable.hpp"
#include "boost/url/url.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

#include "croncpp.h"

#include "utl/verify.h"

#include "nigiri/logging.h"

#include "motis/http_req.h"
#include "motis/import.h"

namespace asio = boost::asio;
namespace fs = std::filesystem;

namespace motis {

namespace {

// A dataset is retried at most this often after a failed reload attempt,
// regardless of how often `run_static_reload`'s minutely check runs -- this
// avoids hammering a misbehaving download URL or repeatedly re-parsing a
// broken feed while a real problem is being fixed upstream.
constexpr auto const kMinRetryBackoff = std::chrono::minutes{5};

}  // namespace

std::vector<std::string> due_static_reloads(
    config::timetable const& t,
    std::map<std::string, std::chrono::system_clock::time_point> const&
        last_fired,
    std::chrono::system_clock::time_point const now) {
  auto due = std::vector<std::string>{};
  for (auto const& [tag, d] : t.datasets_) {
    if (!d.reload_cron_.has_value()) {
      continue;
    }

    auto const last_it = last_fired.find(tag);
    auto const last = last_it == end(last_fired) ? now : last_it->second;

    try {
      auto const cex = cron::make_cron(*d.reload_cron_);
      auto const next = cron::cron_next(cex, last);
      if (next <= now) {
        due.push_back(tag);
      }
    } catch (std::exception const& e) {
      nigiri::log(nigiri::log_lvl::error, "motis.static_reload",
                  "dataset {}: invalid reload_cron \"{}\": {}", tag,
                  *d.reload_cron_, e.what());
    }
  }
  return due;
}

asio::awaitable<void> reload_static_datasets(
    config const& c,
    fs::path data_path,
    std::vector<std::string> dataset_tags) {
  utl::verify(c.timetable_.has_value(),
              "reload_static_datasets: no timetable configured");

  for (auto const& tag : dataset_tags) {
    auto const it = c.timetable_->datasets_.find(tag);
    utl::verify(it != end(c.timetable_->datasets_),
                "reload_static_datasets: unknown dataset {}", tag);
    auto const& d = it->second;
    if (!d.url_.has_value()) {
      continue;
    }

    nigiri::log(nigiri::log_lvl::info, "motis.static_reload",
                "downloading dataset {} from {}", tag, *d.url_);
    auto const res = co_await http_GET(
        boost::urls::url{*d.url_}, d.download_headers_.value_or(headers_t{}),
        std::chrono::seconds{c.timetable_->http_timeout_});

    auto const tmp_path = fs::path{d.path_ + ".download"};
    {
      auto out = std::ofstream{tmp_path, std::ios::binary | std::ios::trunc};
      utl::verify(out.is_open(), "reload_static_datasets: could not create {}",
                  tmp_path);
      out.exceptions(std::ios_base::badbit | std::ios_base::failbit);
      auto const body = get_http_body(res);
      out.write(body.data(), static_cast<std::streamsize>(body.size()));
    }

    auto ec = std::error_code{};
    fs::rename(tmp_path, d.path_, ec);
    utl::verify(!ec, "reload_static_datasets: could not replace {}: {}",
                d.path_, ec.message());
  }

  nigiri::log(nigiri::log_lvl::info, "motis.static_reload",
              "rebuilding static timetable (datasets: {})",
              fmt::join(dataset_tags, ", "));
  import(c, data_path);
  nigiri::log(nigiri::log_lvl::info, "motis.static_reload",
              "static timetable rebuilt successfully");
}

void run_static_reload(asio::io_context& ioc,
                       config const& c,
                       fs::path data_path,
                       std::function<void()> on_reloaded) {
  asio::co_spawn(
      ioc,
      [&c, data_path = std::move(data_path),
       on_reloaded = std::move(on_reloaded)]() -> asio::awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        auto last_fired =
            std::map<std::string, std::chrono::system_clock::time_point>{};
        auto last_attempt =
            std::map<std::string, std::chrono::system_clock::time_point>{};
        auto const start = std::chrono::system_clock::now();
        for (auto const& [tag, d] : c.timetable_->datasets_) {
          if (d.reload_cron_.has_value()) {
            last_fired.emplace(tag, start);
          }
        }

        while (true) {
          auto const now = std::chrono::system_clock::now();
          auto due = due_static_reloads(*c.timetable_, last_fired, now);
          due.erase(std::remove_if(begin(due), end(due),
                                   [&](std::string const& tag) {
                                     auto const it = last_attempt.find(tag);
                                     return it != end(last_attempt) &&
                                            now - it->second < kMinRetryBackoff;
                                   }),
                    end(due));

          if (!due.empty()) {
            for (auto const& tag : due) {
              last_attempt[tag] = now;
            }

            try {
              co_await reload_static_datasets(c, data_path, due);
              for (auto const& tag : due) {
                last_fired[tag] = now;
              }
              on_reloaded();
              nigiri::log(nigiri::log_lvl::info, "motis.static_reload",
                          "now serving reloaded static timetable");
            } catch (std::exception const& e) {
              nigiri::log(
                  nigiri::log_lvl::error, "motis.static_reload",
                  "reload failed, keeping previously loaded timetable: {}",
                  e.what());
            } catch (...) {
              nigiri::log(nigiri::log_lvl::error, "motis.static_reload",
                          "reload failed with unknown error, keeping "
                          "previously loaded timetable");
            }
          }

          timer.expires_at(std::chrono::steady_clock::now() +
                           std::chrono::minutes{1});
          co_await timer.async_wait(
              asio::redirect_error(asio::use_awaitable, ec));
          if (ec == asio::error::operation_aborted) {
            co_return;
          }
        }
      },
      asio::detached);
}

}  // namespace motis
