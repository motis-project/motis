#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/asio/use_awaitable.hpp"
#include "boost/beast/core/flat_buffer.hpp"
#include "boost/beast/http.hpp"

#include "fmt/format.h"

namespace motis::test {

// One request per connection.
struct http_test_server {
  using request_t =
      boost::beast::http::request<boost::beast::http::string_body>;
  using response_t =
      boost::beast::http::response<boost::beast::http::string_body>;
  using handler_t = std::function<response_t(request_t const&)>;

  http_test_server(boost::asio::io_context& ioc, handler_t handler)
      : acceptor_{ioc, {boost::asio::ip::make_address("127.0.0.1"), 0U}},
        handler_{std::move(handler)},
        alive_{std::make_shared<bool>(true)} {
    boost::asio::co_spawn(ioc, accept_loop(), boost::asio::detached);
  }

  ~http_test_server() { *alive_ = false; }

  http_test_server(http_test_server const&) = delete;
  http_test_server& operator=(http_test_server const&) = delete;

  unsigned short port() const { return port_; }

  std::string url(std::string_view const path = "") const {
    return fmt::format("http://127.0.0.1:{}/{}", port_, path);
  }

  void stop() {
    auto ec = boost::system::error_code{};
    acceptor_.close(ec);
  }

  static response_t make_response(
      request_t const& req,
      boost::beast::http::status const status,
      std::string body = "",
      std::vector<std::pair<boost::beast::http::field, std::string>> const&
          headers = {}) {
    auto res = response_t{status, req.version()};
    res.keep_alive(false);
    for (auto const& [k, v] : headers) {
      res.set(k, v);
    }
    res.body() = std::move(body);
    res.prepare_payload();
    return res;
  }

  std::vector<request_t> requests_;

private:
  boost::asio::awaitable<void> accept_loop() {
    auto const alive = alive_;
    while (*alive && acceptor_.is_open()) {
      try {
        auto socket =
            co_await acceptor_.async_accept(boost::asio::use_awaitable);
        if (!*alive) {
          co_return;
        }
        co_await serve(std::move(socket));
      } catch (std::exception const&) {
        co_return;  // acceptor closed
      }
    }
  }

  boost::asio::awaitable<void> serve(boost::asio::ip::tcp::socket socket) {
    namespace http = boost::beast::http;
    try {
      auto buf = boost::beast::flat_buffer{};
      auto req = request_t{};
      co_await http::async_read(socket, buf, req, boost::asio::use_awaitable);
      requests_.push_back(req);
      auto res = handler_(req);
      co_await http::async_write(socket, res, boost::asio::use_awaitable);
      auto ec = boost::system::error_code{};
      socket.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
    } catch (std::exception const&) {
    }
  }

  boost::asio::ip::tcp::acceptor acceptor_;
  handler_t handler_;
  std::shared_ptr<bool> alive_;
  unsigned short port_{acceptor_.local_endpoint().port()};
};

}  // namespace motis::test
