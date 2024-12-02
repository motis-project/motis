#include "motis/scheduler/scheduler_algo.h"

#include "boost/context/detail/prefetch.hpp"
#include "boost/fiber/context.hpp"
#include "boost/fiber/detail/context_spinlock_queue.hpp"
#include "boost/fiber/properties.hpp"
#include "boost/fiber/scheduler.hpp"
#include "boost/fiber/type.hpp"

namespace bf = boost::fibers;

namespace motis {

fiber_props::fiber_props(bf::context* ctx) : fiber_properties{ctx} {}

std::atomic<std::uint32_t> work_stealing::counter_{0};
std::vector<boost::intrusive_ptr<work_stealing> > work_stealing::schedulers_{};

void work_stealing::init_(
    std::uint32_t thread_count,
    std::vector<boost::intrusive_ptr<work_stealing> >& schedulers) {
  std::vector<boost::intrusive_ptr<work_stealing> >{thread_count, nullptr}.swap(
      schedulers);
}

work_stealing::work_stealing(std::uint32_t thread_count, bool suspend)
    : id_{counter_++}, thread_count_{thread_count}, suspend_{suspend} {
  static boost::fibers::detail::thread_barrier b{thread_count};
  // initialize the array of schedulers
  static std::once_flag flag;
  std::call_once(flag, &work_stealing::init_, thread_count_,
                 std::ref(schedulers_));
  // register pointer of this scheduler
  schedulers_[id_] = this;
  b.wait();
}

void work_stealing::awakened(bf::context* ctx, fiber_props& props) noexcept {
  if (!ctx->is_context(bf::type::pinned_context)) {
    ctx->detach();
  }
  if (props.type_ == fiber_props::type::kHighPrio) {
    props.type_ = fiber_props::type::kLowPrio;
    high_prio_rqueue_.push(ctx);
  } else {
    rqueue_.push(ctx);
  }
}

bf::context* work_stealing::pick_next() noexcept {
  bf::context* victim = nullptr;
  if (victim = high_prio_rqueue_.pop(); nullptr != victim) {
    boost::context::detail::prefetch_range(victim, sizeof(bf::context));
    if (!victim->is_context(bf::type::pinned_context)) {
      bf::context::active()->attach(victim);
    }
  } else if (victim = rqueue_.pop(); nullptr != victim) {
    boost::context::detail::prefetch_range(victim, sizeof(bf::context));
    if (!victim->is_context(bf::type::pinned_context)) {
      bf::context::active()->attach(victim);
    }
  } else if (thread_count_ > 1U) {
    std::uint32_t id = 0;
    std::size_t count = 0, size = schedulers_.size();
    static thread_local std::minstd_rand generator{std::random_device{}()};
    std::uniform_int_distribution<std::uint32_t> distribution{
        0, static_cast<std::uint32_t>(thread_count_ - 1)};
    do {
      do {
        ++count;
        // random selection of one logical cpu
        // that belongs to the local NUMA node
        id = distribution(generator);
        // prevent stealing from own scheduler
      } while (id == id_);
      // steal context from other scheduler
      victim = schedulers_[id]->steal();
    } while (nullptr == victim && count < size);
    if (nullptr != victim) {
      boost::context::detail::prefetch_range(victim, sizeof(bf::context));
      BOOST_ASSERT(!victim->is_context(bf::type::pinned_context));
      bf::context::active()->attach(victim);
    }
  }
  return victim;
}

void work_stealing::suspend_until(
    std::chrono::steady_clock::time_point const& time_point) noexcept {
  if (suspend_) {
    if ((std::chrono::steady_clock::time_point::max)() == time_point) {
      std::unique_lock<std::mutex> lk{mtx_};
      cnd_.wait(lk, [this]() { return flag_; });
      flag_ = false;
    } else {
      std::unique_lock<std::mutex> lk{mtx_};
      cnd_.wait_until(lk, time_point, [this]() { return flag_; });
      flag_ = false;
    }
  }
}

void work_stealing::notify() noexcept {
  if (suspend_) {
    std::unique_lock<std::mutex> lk{mtx_};
    flag_ = true;
    lk.unlock();
    cnd_.notify_all();
  }
}

}  // namespace motis