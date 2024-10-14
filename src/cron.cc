#include "motis/cron.h"

#include <iostream>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/asio/this_coro.hpp"
#include "boost/asio/use_awaitable.hpp"

namespace motis {

namespace asio = boost::asio;
using asio::awaitable;
using asio::use_awaitable;
using namespace std::chrono_literals;

awaitable<void> cron(std::chrono::seconds const interval, cron_fn_t f) {
  auto executor = co_await asio::this_coro::executor;
  auto timer = asio::steady_timer{executor};
  auto ec = boost::system::error_code{};
  while (true) {
    try {
      f();
    } catch (std::exception const& e) {
      std::cerr << "EXCEPTION CAUGHT IN CRON: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "EXCEPTION CAUGHT IN CRON" << std::endl;
    }
    timer.expires_after(interval);
    co_await timer.async_wait(asio::redirect_error(use_awaitable, ec));
    if (ec == asio::error::operation_aborted) {
      co_return;
    }
  }
}

void cron(boost::asio::io_context& ioc,
          std::chrono::seconds const interval,
          cron_fn_t f) {
  boost::asio::co_spawn(ioc, cron(interval, std::move(f)),
                        boost::asio::detached);
}

}  // namespace motis