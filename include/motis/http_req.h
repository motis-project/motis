#pragma once

#include <chrono>
#include <map>
#include <string>

#include "boost/asio/awaitable.hpp"
#include "boost/beast/core/flat_buffer.hpp"
#include "boost/beast/http/dynamic_body.hpp"
#include "boost/beast/http/empty_body.hpp"
#include "boost/beast/http/read.hpp"
#include "boost/beast/http/write.hpp"
#include "boost/url/url.hpp"

#include "utl/verify.h"

namespace motis {

namespace beast = boost::beast;
namespace http = beast::http;

using http_response = http::response<boost::beast::http::dynamic_body>;

constexpr auto const kBodySizeLimit = 512U * 1024U * 1024U;  // 512 M

struct proxy {
  bool use_tls_;
  bool use_connect_;
  std::string host_, port_;
};

boost::asio::awaitable<http_response> http_GET(
    boost::urls::url,
    std::map<std::string, std::string> const& headers,
    std::chrono::seconds timeout,
    std::optional<proxy> const& = std::nullopt);

boost::asio::awaitable<http_response> http_POST(
    boost::urls::url,
    std::map<std::string, std::string> const& headers,
    std::string const& body,
    std::chrono::seconds timeout,
    std::optional<proxy> const& = std::nullopt);

template <typename Stream>
boost::asio::awaitable<void> http_CONNECT(Stream& stream,
                                          boost::urls::url const& url,
                                          std::optional<proxy> const& proxy) {
  if (!proxy) {
    co_return;
  }
  auto const target = std::string(url.host()) + ":" +
                      (url.has_port() ? std::string(url.port()) : "443");

  http::request<http::empty_body> req{http::verb::connect, target, 11};
  req.set(http::field::host, target);
  beast::flat_buffer buf;

  co_await http::async_write(beast::get_lowest_layer(stream), req);

  http::response_parser<http::empty_body> res;
  res.skip(true);
  co_await http::async_read_header(beast::get_lowest_layer(stream), buf, res);

  if (res.get().result() != http::status::ok) {
    throw utl::fail("CONNECT failed: target={}, status={}", target,
                    res.get().result_int());
  }
}

std::string get_http_body(http_response const&);

}  // namespace motis
