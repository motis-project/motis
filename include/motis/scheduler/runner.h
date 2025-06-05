#pragma once

#include <atomic>
#include <functional>

#include "boost/fiber/algo/work_stealing.hpp"
#include "boost/fiber/buffered_channel.hpp"
#include "boost/fiber/operations.hpp"

#include "net/web_server/query_router.h"

#include "motis/scheduler/scheduler_algo.h"

namespace motis {

struct runner {
  runner(std::size_t const n_threads, std::size_t const buffer_size)
      : n_threads_{n_threads}, ch_{buffer_size} {}

  auto run_fn() {
    return [&]() {
      boost::fibers::use_scheduling_algorithm<work_stealing>(n_threads_);
      auto t = net::fiber_exec::task_t{};
      while (ch_.pop(t) != boost::fibers::channel_op_status::closed) {
        t();
      }
    };
  }

  std::size_t n_threads_;
  net::fiber_exec::channel_t ch_;
};

}  // namespace motis