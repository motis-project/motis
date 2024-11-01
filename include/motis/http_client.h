#pragma once

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <string>

#include <iostream>

#include "boost/asio/awaitable.hpp"
#include "boost/beast/http/dynamic_body.hpp"
#include "boost/beast/http/message.hpp"
#include "boost/url/url.hpp"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/experimental/awaitable_operators.hpp"
#include "boost/asio/experimental/channel.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/asio/ssl.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/http.hpp"
#include "boost/beast/ssl/ssl_stream.hpp"
#include "boost/beast/version.hpp"

#include "utl/verify.h"

#include "motis/http_req.h"
#include "motis/types.h"

namespace motis {

struct http_client {
  struct request {
    template <typename Executor>
    request(boost::urls::url&& url,
            std::map<std::string, std::string>&& headers,
            Executor const& executor)
        : url_{std::move(url)},
          headers_{std::move(headers)},
          response_channel_{executor} {}

    boost::urls::url url_{};
    std::map<std::string, std::string> headers_{};
    boost::asio::experimental::channel<void(
        boost::system::error_code,
        boost::beast::http::response<boost::beast::http::dynamic_body>)>
        response_channel_;
  };

  struct connection_key {
    friend bool operator==(connection_key const&,
                           connection_key const&) = default;

    std::string host_;
    std::string port_;
    bool ssl_{};
  };

  struct connection : public std::enable_shared_from_this<connection> {
    template <typename Executor>
    connection(connection_key const& key, Executor const& executor)
        : key_{key}, request_channel_{executor} {}

    connection_key key_{};
    std::unique_ptr<boost::beast::tcp_stream> stream_;
    std::unique_ptr<boost::asio::ssl::stream<boost::beast::tcp_stream>>
        ssl_stream_;
    boost::asio::experimental::channel<void(boost::system::error_code,
                                            std::shared_ptr<request>)>
        request_channel_;
    std::deque<std::shared_ptr<request>> pending_requests_;

    unsigned n_sent_{};
    unsigned n_received_{};
  };

  http_client() {
    std::cout << "[http] client ctor" << std::endl;
    ssl_ctx_.set_default_verify_paths();
    ssl_ctx_.set_verify_mode(boost::asio::ssl::verify_none);
    ssl_ctx_.set_options(boost::asio::ssl::context::default_workarounds |
                         boost::asio::ssl::context::no_sslv2 |
                         boost::asio::ssl::context::no_sslv3 |
                         boost::asio::ssl::context::single_dh_use);
  }

  ~http_client() {
    std::cout << "[http] client dtor" << std::endl;
    for (auto const& [key, conn] : connections_) {
      close(conn);
    }
    std::cout << "[http] client dtor END" << std::endl;
  }

  boost::asio::awaitable<void> timeout(
      std::chrono::steady_clock::duration duration) {
    boost::asio::steady_timer timer(co_await boost::asio::this_coro::executor);
    timer.expires_after(duration);
    co_await timer.async_wait();
  }

  void close(std::shared_ptr<connection> conn) {
    std::cout << "[http] close(" << conn->key_.host_ << ":" << conn->key_.port_
              << ", ssl=" << conn->key_.ssl_ << ", s=" << conn->n_sent_
              << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << ")" << std::endl;

    if (conn->key_.ssl_) {
      auto ec = boost::beast::error_code{};
      boost::beast::get_lowest_layer(*conn->ssl_stream_)
          .socket()
          .shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    } else {
      auto ec = boost::beast::error_code{};
      boost::beast::get_lowest_layer(*conn->stream_)
          .socket()
          .shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    }
  }

  boost::asio::awaitable<void> connect(std::shared_ptr<connection> conn) {
    std::cout << "[http] connect(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << ")" << std::endl;
    auto executor = co_await boost::asio::this_coro::executor;
    auto resolver = boost::asio::ip::tcp::resolver{executor};

    auto const results =
        co_await resolver.async_resolve(conn->key_.host_, conn->key_.port_);
    if (conn->key_.ssl_) {
      std::cout << "[http] connect(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): creating ssl stream" << std::endl;
      conn->ssl_stream_ =
          std::make_unique<boost::asio::ssl::stream<boost::beast::tcp_stream>>(
              executor, ssl_ctx_);

      if (!SSL_set_tlsext_host_name(
              conn->ssl_stream_->native_handle(),
              const_cast<char*>(conn->key_.host_.c_str()))) {
        throw boost::system::system_error{
            {static_cast<int>(::ERR_get_error()),
             boost::asio::error::get_ssl_category()}};
      }

      co_await boost::beast::get_lowest_layer(*conn->ssl_stream_)
          .async_connect(results);
      co_await conn->ssl_stream_->async_handshake(
          boost::asio::ssl::stream_base::client);
      std::cout << "[http] connect(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): ssl stream established" << std::endl;
    } else {
      std::cout << "[http] connect(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): creating tcp stream" << std::endl;
      conn->stream_ = std::make_unique<boost::beast::tcp_stream>(executor);
      co_await conn->stream_->async_connect(results);
      std::cout << "[http] connect(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): tcp stream established" << std::endl;
    }
  }

  boost::asio::awaitable<void> send_requests(std::shared_ptr<connection> conn) {
    std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << ")" << std::endl;
    try {
      auto const send_request = [&](std::shared_ptr<request> request)
          -> boost::asio::awaitable<void> {
        std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): got request: " << request->url_.encoded_target()
                  << std::endl;

        auto req = boost::beast::http::request<boost::beast::http::string_body>{
            boost::beast::http::verb::get, request->url_.encoded_target(), 11};
        req.set(boost::beast::http::field::host, request->url_.host());
        req.set(boost::beast::http::field::user_agent,
                BOOST_BEAST_VERSION_STRING);
        req.set(boost::beast::http::field::accept_encoding, "gzip");
        for (auto const& [k, v] : request->headers_) {
          req.set(k, v);
        }
        req.keep_alive(true);

        if (conn->key_.ssl_) {
          co_await boost::beast::http::async_write(*conn->ssl_stream_, req);
        } else {
          co_await boost::beast::http::async_write(*conn->stream_, req);
        }
        ++conn->n_sent_;
        std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): request sent" << std::endl;
      };

      if (!conn->pending_requests_.empty()) {
        std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): processing pending requests after reconnect"
                  << std::endl;
        for (auto const& request : conn->pending_requests_) {
          co_await send_request(request);
        }
      }

      while (conn->request_channel_.is_open()) {
        std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): waiting for request" << std::endl;
        auto request = co_await conn->request_channel_.async_receive();
        conn->pending_requests_.push_back(request);
        co_await send_request(request);
      }
    } catch (std::exception const& ex) {
      std::cout << "[http] exception in send_requests: " << ex.what()
                << std::endl;
      // throw ex;
    }
    std::cout << "[http] send_requests(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << "): END"
              << std::endl;
  }

  boost::asio::awaitable<void> receive_responses(
      std::shared_ptr<connection> conn) {
    std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << ")" << std::endl;
    try {
      for (;;) {
        auto buffer = boost::beast::flat_buffer{};
        auto res =
            boost::beast::http::response<boost::beast::http::dynamic_body>{};

        std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): waiting for response" << std::endl;
        if (conn->key_.ssl_) {
          co_await boost::beast::http::async_read(*conn->ssl_stream_, buffer,
                                                  res);
        } else {
          co_await boost::beast::http::async_read(*conn->stream_, buffer, res);
        }
        ++conn->n_received_;

        std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): response received, connection header: "
                  << res[boost::beast::http::field::connection] << std::endl;

        utl::verify(!conn->pending_requests_.empty(),
                    "received response without pending request");
        auto req = conn->pending_requests_.front();
        conn->pending_requests_.pop_front();

        std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): finishing pending request" << std::endl;
        co_await req->response_channel_.async_send(boost::system::error_code{},
                                                   std::move(res));
        std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
                  << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                  << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                  << ", p=" << conn->pending_requests_.size()
                  << "): pending request finished" << std::endl;
      }
    } catch (std::exception const& ex) {
      std::cout << "[http] exception in receive_responses: " << ex.what()
                << std::endl;
      // throw ex;
    }
    // TODO: boost::system::system_error
    std::cout << "[http] receive_response(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << "): END"
              << std::endl;
  }

  boost::asio::awaitable<void> handle_connection(
      std::shared_ptr<connection> conn) {
    using namespace boost::asio::experimental::awaitable_operators;
    std::cout << "[http] handle_connection(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << ")" << std::endl;
    do {
      std::cout << "[http] handle_connection(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): (re)connecting" << std::endl;
      co_await connect(conn);
      std::cout << "[http] handle_connection(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): connected, running send_requests + receive_responses"
                << std::endl;
      co_await (receive_responses(conn) || send_requests(conn));
      std::cout << "[http] handle_connection(" << conn->key_.host_ << ":"
                << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
                << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
                << ", p=" << conn->pending_requests_.size()
                << "): send / receive done" << std::endl;
    } while (!conn->pending_requests_.empty());
    std::cout << "[http] handle_connection(" << conn->key_.host_ << ":"
              << conn->key_.port_ << ", ssl=" << conn->key_.ssl_
              << ", s=" << conn->n_sent_ << ", r=" << conn->n_received_
              << ", p=" << conn->pending_requests_.size() << "): END"
              << std::endl;
  }

  boost::asio::awaitable<http_response> get(
      boost::urls::url url, std::map<std::string, std::string> headers) {
    auto const https = url.scheme_id() == boost::urls::scheme::https;
    auto const key = connection_key{
        url.host(), url.has_port() ? url.port() : (https ? "443" : "80"),
        https};

    std::cout << "[http] get(" << key.host_ << ":" << key.port_
              << ", ssl=" << key.ssl_ << "), target=" << url.encoded_target()
              << std::endl;

    auto executor = co_await boost::asio::this_coro::executor;
    if (auto const it = connections_.find(key); it == connections_.end()) {
      std::cout << "[http] get(" << key.host_ << ":" << key.port_
                << ", ssl=" << key.ssl_
                << "): connection not found, spawning new connection"
                << std::endl;
      auto conn = std::make_shared<connection>(key, executor);
      connections_[key] = conn;

      boost::asio::co_spawn(executor, handle_connection(conn),
                            boost::asio::detached);

      std::cout << "[http] get(" << key.host_ << ":" << key.port_
                << ", ssl=" << key.ssl_ << "): connection spawned" << std::endl;
    } else {
      std::cout << "[http] get(" << key.host_ << ":" << key.port_
                << ", ssl=" << key.ssl_ << "): existing connection found"
                << std::endl;
    }

    auto req =
        std::make_shared<request>(std::move(url), std::move(headers), executor);
    std::cout << "[http] get(" << key.host_ << ":" << key.port_
              << ", ssl=" << key.ssl_ << "): writing request" << std::endl;
    co_await connections_[key]->request_channel_.async_send(
        boost::system::error_code{}, req);
    std::cout << "[http] get(" << key.host_ << ":" << key.port_
              << ", ssl=" << key.ssl_ << "): waiting for response" << std::endl;
    auto response = co_await req->response_channel_.async_receive();
    std::cout << "[http] get(" << key.host_ << ":" << key.port_
              << ", ssl=" << key.ssl_ << "): response received" << std::endl;
    co_return response;
  }

  hash_map<connection_key, std::shared_ptr<connection>> connections_;
  std::chrono::seconds timeout_{std::chrono::seconds{10}};
  boost::asio::ssl::context ssl_ctx_{boost::asio::ssl::context::tlsv12_client};
};

}  // namespace motis
