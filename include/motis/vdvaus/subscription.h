#pragma once

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::vdvaus {

void subscription(boost::asio::io_context&, config const&, data&);

void shutdown(boost::asio::io_context&, config const&, data&);

}  // namespace motis::vdvaus