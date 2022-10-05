#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "boost/asio/io_context.hpp"

#if defined(NET_TLS)
#include "boost/asio/ssl/context.hpp"
#endif

#include "boost/beast/http/buffer_body.hpp"
#include "boost/beast/http/empty_body.hpp"
#include "boost/beast/http/file_body.hpp"
#include "boost/beast/http/message.hpp"
#include "boost/beast/http/string_body.hpp"
#include "net/web_server/web_server.h"

namespace net {

using ws_session_ptr = std::weak_ptr<ws_session>;

struct test_server {
  using http_req_t =
      boost::beast::http::request<boost::beast::http::string_body>;
  using string_res_t =
      boost::beast::http::response<boost::beast::http::string_body>;
  using buffer_res_t =
      boost::beast::http::response<boost::beast::http::buffer_body>;
  using file_res_t =
      boost::beast::http::response<boost::beast::http::file_body>;
  using empty_res_t =
      boost::beast::http::response<boost::beast::http::empty_body>;
  using http_res_t =
      std::variant<string_res_t, buffer_res_t, file_res_t, empty_res_t>;

  using http_res_cb_t = std::function<void(http_res_t&&)>;
  using http_req_cb_t = std::function<void(http_req_t, http_res_cb_t, bool)>;

  using ws_msg_cb_t =
      std::function<void(ws_session_ptr, std::string const&, ws_msg_type)>;
  using ws_open_cb_t = std::function<void(
      ws_session_ptr, std::string const& /* target */, bool /* is SSL */)>;
  using ws_close_cb_t = std::function<void(void*)>;
  using ws_upgrade_ok_cb_t = std::function<bool(http_req_t const&)>;

#if defined(NET_TLS)
  explicit test_server(boost::asio::io_context&, boost::asio::ssl::context&);
#else
  explicit test_server(boost::asio::io_context&);
#endif
  ~test_server();

  test_server(test_server&&) = default;
  test_server& operator=(test_server&&) = default;

  test_server(test_server const&) = delete;
  test_server& operator=(test_server const&) = delete;

  void init(std::string const&, std::string const&,
            boost::system::error_code&) const;
  void run() const;
  void stop() const;

  void set_timeout(std::chrono::seconds const&) const;
  void set_request_body_limit(std::uint64_t) const;
  void set_request_queue_limit(std::size_t) const;

  void on_http_request(http_req_cb_t) const;
  void on_ws_msg(ws_msg_cb_t) const;
  void on_ws_open(ws_open_cb_t) const;
  void on_ws_close(ws_close_cb_t) const;
  void on_upgrade_ok(ws_upgrade_ok_cb_t) const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace net
