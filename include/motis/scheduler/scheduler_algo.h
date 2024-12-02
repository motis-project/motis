#pragma once

#include <cinttypes>
#include <vector>

#include "boost/context/detail/prefetch.hpp"
#include "boost/fiber/algo/algorithm.hpp"
#include "boost/fiber/detail/context_spinlock_queue.hpp"
#include "boost/fiber/detail/thread_barrier.hpp"
#include "boost/fiber/properties.hpp"
#include "boost/fiber/scheduler.hpp"
#include "boost/intrusive_ptr.hpp"

namespace motis {

struct fiber_props : public boost::fibers::fiber_properties {
  fiber_props(boost::fibers::context*);
  ~fiber_props() override;

  // In order to keep request latency low, finishing already started requests
  // has to be prioritized over new requests. Otherwise, the server only starts
  // new requests and never finishes anything.
  enum class type : std::uint8_t {
    kHighPrio,  // follow-up work scheduled by work
    kLowPrio  // initial work scheduled by I/O (web request / batch query)
  } type_{type::kHighPrio};
};

struct work_stealing
    : public boost::fibers::algo::algorithm_with_properties<fiber_props> {
  static std::atomic<std::uint32_t> counter_;
  static std::vector<boost::intrusive_ptr<work_stealing> > schedulers_;

  std::uint32_t id_;
  std::uint32_t thread_count_;
  boost::fibers::detail::context_spinlock_queue rqueue_{};
  boost::fibers::detail::context_spinlock_queue high_prio_rqueue_{};
  std::mutex mtx_{};
  std::condition_variable cnd_{};
  bool flag_{false};
  bool suspend_;

  static void init_(std::uint32_t,
                    std::vector<boost::intrusive_ptr<work_stealing> >&);

  work_stealing(std::uint32_t, bool = false);

  work_stealing(work_stealing const&) = delete;
  work_stealing(work_stealing&&) = delete;

  work_stealing& operator=(work_stealing const&) = delete;
  work_stealing& operator=(work_stealing&&) = delete;

  void awakened(boost::fibers::context*, fiber_props&) noexcept override;

  boost::fibers::context* pick_next() noexcept override;

  virtual boost::fibers::context* steal() noexcept { return rqueue_.steal(); }

  bool has_ready_fibers() const noexcept override { return !rqueue_.empty(); }

  void suspend_until(
      std::chrono::steady_clock::time_point const&) noexcept override;

  void notify() noexcept override;
};

}  // namespace motis