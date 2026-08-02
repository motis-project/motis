#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis {

struct rt_update_hooks {
  std::function<std::chrono::system_clock::time_point()> now_;
  std::function<void(std::size_t endpoint_idx, bool fallback)>
      after_gtfsrt_apply_;
};

void run_rt_update(boost::asio::io_context&,
                   config const&,
                   data&,
                   rt_update_hooks = {});

void apply_canned_rt_update(config const&, data&);

}  // namespace motis
