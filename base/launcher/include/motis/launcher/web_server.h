#pragma once

#include <memory>
#include <string>

#include "boost/asio/io_service.hpp"

#include "motis/module/controller.h"

namespace motis::launcher {

struct web_server {
  web_server(boost::asio::io_service&, motis::module::controller&);
  ~web_server();

  web_server(web_server&&) = default;
  web_server& operator=(web_server&&) = default;

  web_server(web_server const&) = delete;
  web_server& operator=(web_server const&) = delete;

  void listen(std::string const& host, std::string const& port,
#if defined(NET_TLS)
              std::string const& cert_path, std::string const& priv_key_path,
              std::string const& dh_path,
#endif
              std::string const& log_path, std::string const& static_path,
              boost::system::error_code& ec);
  void stop();

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::launcher
