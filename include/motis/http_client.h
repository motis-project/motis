#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "boost/asio/awaitable.hpp"
#include "boost/url/url.hpp"

#include "motis/http_req.h"
#include "motis/types.h"

namespace motis {

constexpr auto const kUnlimitedHttpPipelining =
    std::numeric_limits<std::size_t>::max();

struct http_client {
  struct connection_key {
    friend bool operator==(connection_key const&,
                           connection_key const&) = default;

    std::string host_;
    std::string port_;
    bool ssl_{};
  };

  struct connection;
  struct request;

  ~http_client();

  boost::asio::awaitable<http_response> get(
      boost::urls::url url, std::map<std::string, std::string> headers);

  hash_map<connection_key, std::shared_ptr<connection>> connections_;
  std::chrono::seconds timeout_{std::chrono::seconds{10}};
};

}  // namespace motis
