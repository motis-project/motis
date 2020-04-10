#pragma once

#include <algorithm>
#include <memory>
#include <thread>

#include "boost/asio/io_service.hpp"
#include "boost/bind.hpp"

namespace motis::parking::prepare {

struct thread_pool {
  explicit thread_pool(
      unsigned num_threads = std::thread::hardware_concurrency())
      : ios_(),
        work_(std::make_unique<boost::asio::io_service::work>(ios_)),
        threads_(num_threads) {
    for (auto& t : threads_) {
      t = std::thread(boost::bind(&boost::asio::io_service::run, &ios_));
    }
  }

  template <typename Handler>
  void post(Handler handler) {
    ios_.post(handler);
  }

  void join() {
    work_.reset(nullptr);
    std::for_each(begin(threads_), end(threads_),
                  [](std::thread& t) { t.join(); });
  }

private:
  boost::asio::io_service ios_;
  std::unique_ptr<boost::asio::io_service::work> work_;
  std::vector<std::thread> threads_;
};

}  // namespace motis::parking::prepare
