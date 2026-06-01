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

template <typename Fn>
boost::asio::awaitable<void> repeat(std::chrono::seconds const interval,
                                    std::string_view const task_name,
                                    Fn&& fn) {
  namespace asio = boost::asio;
  auto executor = co_await asio::this_coro::executor;
  auto timer = asio::steady_timer{executor};
  auto ec = boost::system::error_code{};
  while (true) {
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
