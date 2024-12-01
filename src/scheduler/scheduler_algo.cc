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

scheduler_algo::scheduler_algo(boost::fibers::detail::thread_barrier& b,
                               std::vector<scheduler_algo*>& schedulers,
                               std::uint32_t const id)
    : schedulers_{schedulers}, id_{id} {
  schedulers_[id] = this;
  b.wait();
}

void scheduler_algo::awakened(bf::context* ctx, fiber_props& props) noexcept {
  if (!ctx->is_context(bf::type::pinned_context)) {
    ctx->detach();
  }
  auto const orig_type = props.type_;
  props.type_ = fiber_props::type::kWork;  // Continuations are prioritized.
  orig_type == fiber_props::type::kWork ? work_queue_.push(ctx)
                                        : io_queue_.push(ctx);
}

bf::context* scheduler_algo::pick_next() noexcept {
  using boost::context::detail::prefetch_range;
  bf::context* victim = nullptr;
  if (victim = work_queue_.pop(); victim != nullptr) {
    // Highest priority: work continuation.
    prefetch_range(victim, sizeof(bf::context));
    if (!victim->is_context(bf::type::pinned_context)) {
      bf::context::active()->attach(victim);
    }
  } else if (victim = io_queue_.pop(); victim != nullptr) {
    // Lower priority: I/O from our own queue.
    prefetch_range(victim, sizeof(bf::context));
    if (!victim->is_context(bf::type::pinned_context)) {
      bf::context::active()->attach(victim);
    }
  } else {  // Fallback: try to steal from another thread.
    auto id = 0U;
    auto count = std::size_t{0U};
    auto size = schedulers_.size();
    static thread_local std::minstd_rand generator{std::random_device{}()};
    auto distribution = std::uniform_int_distribution<std::uint32_t>{
        0, static_cast<std::uint32_t>(size - 1)};

    do {
      do {
        ++count;
        id = distribution(generator);
      } while (id == id_ /* don't steal from own scheduler */);
      victim = schedulers_[id]->steal();
    } while (victim == nullptr && count < size);

    if (victim != nullptr) {
      prefetch_range(victim, sizeof(bf::context));
      BOOST_ASSERT(!victim->is_context(bf::type::pinned_context));
      bf::context::active()->attach(victim);
    }
  }
  return victim;
}

bool scheduler_algo::has_ready_fibers() const noexcept {
  return !(work_queue_.empty() && io_queue_.empty());
}

bf::context* scheduler_algo::steal() noexcept {
  auto work = work_queue_.pop();
  if (work != nullptr) {
    return work;
  }
  return io_queue_.pop();
}

void scheduler_algo::suspend_until(
    std::chrono::steady_clock::time_point const& time_point) noexcept {
  if (suspend_) {
    if ((std::chrono::steady_clock::time_point::max)() == time_point) {
      auto lk = std::unique_lock<std::mutex>{mtx_};
      cnd_.wait(lk, [this]() { return flag_; });
      flag_ = false;
    } else {
      auto lk = std::unique_lock<std::mutex>{mtx_};
      cnd_.wait_until(lk, time_point, [this]() { return flag_; });
      flag_ = false;
    }
  }
}

void scheduler_algo::notify() noexcept {
  if (suspend_) {
    auto lk = std::unique_lock<std::mutex>{mtx_};
    flag_ = true;
    lk.unlock();
    cnd_.notify_all();
  }
}

}  // namespace motis