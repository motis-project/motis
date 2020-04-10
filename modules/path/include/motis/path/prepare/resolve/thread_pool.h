#pragma once

#include <condition_variable>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace motis::path {

struct thread_pool {
  explicit thread_pool(
      std::function<void()> initializer = [] {},
      std::function<void()> finalizer = [] {})
      : initializer_{std::move(initializer)},
        finalizer_{std::move(finalizer)},
        threads_active_{0},
        stop_{false},
        job_count_{0},
        counter_{0} {
    for (auto i = 0U; i < std::thread::hardware_concurrency(); ++i) {
      threads_.emplace_back([&, this] {
        initializer_();
        while (true) {
          {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [&] { return stop_ || counter_ < job_count_; });
            if (stop_) {
              break;
            }

            ++threads_active_;
          }

          while (true) {
            auto const idx = counter_.fetch_add(1);
            if (idx >= job_count_) {
              break;
            }
            fn_(idx);
          }

          {
            std::unique_lock<std::mutex> lk(mutex_);
            --threads_active_;
            cv_.notify_all();
          }
        }
        finalizer_();
      });
    }
  }

  ~thread_pool() {
    stop_ = true;
    cv_.notify_all();
    std::for_each(begin(threads_), end(threads_), [](auto& t) { t.join(); });
  }

  thread_pool(thread_pool const&) noexcept = delete;  // NOLINT
  thread_pool& operator=(thread_pool const&) noexcept = delete;  // NOLINT
  thread_pool(thread_pool&&) noexcept = delete;  // NOLINT
  thread_pool& operator=(thread_pool&&) noexcept = delete;  // NOLINT

  void execute(size_t const job_count, std::function<void(size_t)>&& fn) {
    if (job_count == 0) {
      return;
    }

    counter_ = 0;
    job_count_ = job_count;
    fn_ = fn;

    cv_.notify_all();

    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk,
             [&] { return threads_active_ == 0 && counter_ >= job_count_; });
  }

private:
  std::function<void()> initializer_, finalizer_;

  std::vector<std::thread> threads_;
  std::atomic_size_t threads_active_;
  bool stop_;

  std::mutex mutex_;
  std::condition_variable cv_;

  size_t job_count_;
  std::atomic_size_t counter_;
  std::function<void(size_t)> fn_;
};

}  // namespace motis::path
