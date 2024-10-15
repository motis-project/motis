#pragma once

#include <functional>
#include <memory>

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_service.hpp"

#include "ctx/access_request.h"

namespace motis::module {

struct dispatcher;

struct timer : public std::enable_shared_from_this<timer> {
  timer(char const* name, dispatcher*,
        boost::posix_time::time_duration interval, std::function<void()> fn,
        ctx::accesses_t&& access);

  void stop();
  void schedule();
  void exec(boost::system::error_code const& ec);

  std::atomic_bool stopped_{false};
  char const* name_;
  boost::posix_time::time_duration interval_;
  boost::asio::deadline_timer timer_;
  std::function<void()> fn_;
  dispatcher* dispatcher_;
  ctx::accesses_t access_;
};

}  // namespace motis::module