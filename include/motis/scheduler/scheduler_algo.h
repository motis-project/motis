#pragma once

#include <cinttypes>
#include <vector>

#include "boost/context/detail/prefetch.hpp"
#include "boost/fiber/algo/algorithm.hpp"
#include "boost/fiber/detail/context_spinlock_queue.hpp"
#include "boost/fiber/detail/thread_barrier.hpp"
#include "boost/fiber/properties.hpp"
#include "boost/fiber/scheduler.hpp"

namespace motis {

struct fiber_props : public boost::fibers::fiber_properties {
  fiber_props(boost::fibers::context*);

  // In order to keep request latency low, finishing already started requests
  // has to be prioritized over new requests. Otherwise, the server only starts
  // new requests and never finishes anything.
  enum class type : std::uint8_t {
    kWork,  // follow-up work scheduled by work or I/O
    kIo  // initial work scheduled by I/O (web request / batch query)
  } type_{type::kIo};
};

struct scheduler_algo
    : public boost::fibers::algo::algorithm_with_properties<fiber_props> {
  using ready_queue_t = boost::fibers::scheduler::ready_queue_type;

  scheduler_algo(boost::fibers::detail::thread_barrier&,
                 std::vector<scheduler_algo*>& schedulers,
                 std::uint32_t id);

  boost::fibers::context* steal() noexcept;

  virtual void awakened(boost::fibers::context* ctx,
                        fiber_props& props) noexcept override;
  virtual boost::fibers::context* pick_next() noexcept override;
  virtual bool has_ready_fibers() const noexcept override;
  virtual void suspend_until(
      std::chrono::steady_clock::time_point const&) noexcept override;
  virtual void notify() noexcept override;

  std::vector<scheduler_algo*>& schedulers_;
  bool suspend_{false};
  std::uint32_t id_;
  boost::fibers::detail::context_spinlock_queue work_queue_, io_queue_;
  std::mutex mtx_{};
  std::condition_variable cnd_{};
  bool flag_{false};
};

}  // namespace motis