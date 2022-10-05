#pragma once

#include "boost/asio/io_service.hpp"
#include "test_server.h"
namespace motis::intermodal {

struct test_server {
  explicit test_server(boost::asio::io_service&);
  ~test_server();

  test_server(test_server&&) = default;
  test_server& operator = (test_server&&) = default;

  test_server(test_server const&) = delete;
  test_server& operator = (test_server const&) = delete;

  void listen_tome(std::string const&, std::string const&,
                   boost::system::error_code&);

  void stop_it();

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};


} // namespace motis::intermodal
