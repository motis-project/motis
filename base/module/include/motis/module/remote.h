#pragma once

#include <functional>
#include <memory>
#include <string>

#include "boost/asio/io_service.hpp"

#include "motis/module/message.h"
#include "motis/module/registry.h"

namespace motis::module {

struct remote : std::enable_shared_from_this<remote> {
  remote(registry&, boost::asio::io_service&,  //
         std::string const& host, std::string const& port,  //
         std::function<void()> const& on_register = nullptr,
         std::function<void()> const& on_unregister = nullptr);

  void send(msg_ptr const&, callback) const;
  void stop() const;
  void start() const;

  struct impl;
  std::shared_ptr<impl> impl_;
};

}  // namespace motis::module