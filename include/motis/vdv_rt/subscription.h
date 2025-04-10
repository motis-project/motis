#pragma once

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

void unsubscribe(boost::asio::io_context&, config const&, data&);

void subscription(boost::asio::io_context&, config const&, data&);

}  // namespace motis::vdv_rt