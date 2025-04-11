#pragma once

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

boost::asio::awaitable<void> unsubscribe(boost::asio::io_context&,
                                         config const&,
                                         data&);

void renew_subscription(boost::asio::io_context&, config const&, data&);

}  // namespace motis::vdv_rt