#pragma once

#include <chrono>
#include <functional>
#include <memory>

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis {

struct rt_update_hooks {
  std::function<std::chrono::system_clock::time_point()> now_;
};

void run_rt_update(boost::asio::io_context&,
                   config const&,
                   data&,
                   rt_update_hooks = {});

}  // namespace motis
