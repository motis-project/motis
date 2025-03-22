#pragma once

#include <chrono>
#include <memory>

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis {

void run_rt_update(boost::asio::io_context&, config const&, data&);

}