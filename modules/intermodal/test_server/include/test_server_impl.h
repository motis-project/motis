#pragma once

#include "boost/asio/io_service.hpp"

using namespace std;

namespace motis::intermodal {

struct test_server {
  test_server(boost::asio::io_service&);
  ~test_server();

  test_server(test_server&&) = default;
  test_server& operator = (test_server&&) = default;

  test_server(test_server const&) = delete;
  test_server& operator = (test_server const&) = delete;

  void listen_tome(std::string const& host, std::string const& port,
                   boost::system::error_code& ec);

  void stop_it();

private:
  struct impl;
  unique_ptr<impl> impl_;
};


} // namespace motis::intermodal
