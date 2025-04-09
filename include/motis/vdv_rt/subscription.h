#pragma once

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

void subscription(boost::asio::io_context&, config const&, data& d);

}  // namespace motis::vdv_rt