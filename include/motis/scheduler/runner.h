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
      : init_barrier_{n_threads}, schedulers_{n_threads}, ch_{buffer_size} {}

  auto run_fn() {
    return [&]() {
      /*
    boost::fibers::use_scheduling_algorithm<
    boost::fibers::algo::work_stealing>(schedulers_.size());
    */
      boost::fibers::use_scheduling_algorithm<scheduler_algo>(
          init_barrier_, schedulers_, ++next_id_);
      auto t = net::fiber_exec::task_t{};
      while (ch_.pop(t) != boost::fibers::channel_op_status::closed) {
        t();
      }
    };
  }

  boost::fibers::detail::thread_barrier init_barrier_;
  std::atomic_uint32_t next_id_{0U};
  std::vector<scheduler_algo*> schedulers_;
  net::fiber_exec::channel_t ch_;
};

}  // namespace motis