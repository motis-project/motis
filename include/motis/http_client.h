#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "boost/asio/awaitable.hpp"
#include "boost/system.hpp"
#include "boost/url/url.hpp"

#include "motis/http_req.h"
#include "motis/types.h"

namespace motis {

constexpr auto const kUnlimitedHttpPipelining =
    std::numeric_limits<std::size_t>::max();

enum request_method { GET, POST };

struct http_client {
  enum class error { success = 0, too_many_redirects, request_failed };

  struct error_category_impl : public boost::system::error_category {
    virtual ~error_category_impl() = default;
    char const* name() const noexcept override { return "http_client"; }
    std::string message(int ev) const override;
  };

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

  boost::asio::awaitable<http_response> post(
      boost::urls::url,
      std::map<std::string, std::string> headers,
      std::string body);

  hash_map<connection_key, std::shared_ptr<connection>> connections_;
  std::chrono::seconds timeout_{std::chrono::seconds{10}};
};

}  // namespace motis

namespace boost {
namespace system {

template <>
struct is_error_code_enum<::motis::http_client::error> : std::true_type {};

}  // namespace system
}  // namespace boost

namespace std {

template <>
struct is_error_code_enum<::motis::http_client::error> : std::true_type {};

}  // namespace std
