#pragma once

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::vdv_rt {

void subscribe(boost::asio::io_context&, config const&, vdv_rt::connection&);

}  // namespace motis::vdv_rt