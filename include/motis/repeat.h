#pragma once

#include <chrono>
#include <string>
#include <string_view>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/error.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/asio/this_coro.hpp"
#include "boost/asio/use_awaitable.hpp"

#include "utl/timer.h"

#include "nigiri/logging.h"

namespace motis {

// Runs `fn` repeatedly, with `interval` between the start of consecutive runs,
// timing each run via utl::scoped_timer under `task_name`.
//
// Any exception thrown by `fn` is caught and logged; the loop then continues
// with the next scheduled run. This is meant to be co_spawn'd detached: an
// exception escaping the loop would otherwise be silently swallowed by asio
// (boost::asio::detached's handler ignores it) and the loop would die without
// a trace. Ends cleanly when the io_context is stopped (the timer wait
// completes with operation_aborted).
template <typename Fn>
boost::asio::awaitable<void> repeat(std::chrono::seconds const interval,
                                    std::string_view const task_name,
                                    Fn&& fn) {
  namespace asio = boost::asio;
  auto executor = co_await asio::this_coro::executor;
  auto timer = asio::steady_timer{executor};
  auto ec = boost::system::error_code{};
  while (true) {
    // Remember when we started, so we can schedule the next run.
    auto const start = std::chrono::steady_clock::now();

    try {
      auto const t = utl::scoped_timer{std::string{task_name}};
      co_await fn();
    } catch (std::exception const& e) {
      nigiri::log(nigiri::log_lvl::error, "motis", "{} failed: {}", task_name,
                  e.what());
    } catch (...) {
      nigiri::log(nigiri::log_lvl::error, "motis", "{} failed: unknown error",
                  task_name);
    }

    timer.expires_at(start + interval);
    co_await timer.async_wait(asio::redirect_error(asio::use_awaitable, ec));
    if (ec == asio::error::operation_aborted) {
      co_return;
    }
  }
}

}  // namespace motis
