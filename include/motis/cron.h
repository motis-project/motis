#pragma once

#include <functional>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

namespace motis {

using cron_fn_t = std::function<void()>;

boost::asio::awaitable<void> cron(std::chrono::seconds interval, cron_fn_t);

void cron(boost::asio::io_context&, std::chrono::seconds interval, cron_fn_t);

}  // namespace motis