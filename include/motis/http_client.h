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
    connection(connection_key key, Executor const& executor)
        : key_{std::move(key)}, request_channel_{executor} {
      ssl_ctx_.set_default_verify_paths();
      ssl_ctx_.set_verify_mode(boost::asio::ssl::verify_none);
      ssl_ctx_.set_options(boost::asio::ssl::context::default_workarounds |
                           boost::asio::ssl::context::no_sslv2 |
                           boost::asio::ssl::context::no_sslv3 |
                           boost::asio::ssl::context::single_dh_use);
    }

    ~connection() {
      std::cout << "[http] connection dtor(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;
    }

    void close() {
      std::cout << "[http] close(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;

      if (key_.ssl_) {
        auto ec = boost::beast::error_code{};
        boost::beast::get_lowest_layer(*ssl_stream_)
            .socket()
            .shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
      } else {
        auto ec = boost::beast::error_code{};
        boost::beast::get_lowest_layer(*stream_).socket().shutdown(
            boost::asio::ip::tcp::socket::shutdown_both, ec);
      }
    }

    boost::asio::awaitable<void> connect() {
      std::cout << "[http] connect(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;
      auto executor = co_await boost::asio::this_coro::executor;
      auto resolver = boost::asio::ip::tcp::resolver{executor};

      auto const results =
          co_await resolver.async_resolve(key_.host_, key_.port_);
      if (key_.ssl_) {
        std::cout << "[http] connect(" << key_.host_ << ":" << key_.port_
                  << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): creating ssl stream" << std::endl;
        ssl_stream_ = std::make_unique<
            boost::asio::ssl::stream<boost::beast::tcp_stream>>(executor,
                                                                ssl_ctx_);

        if (!SSL_set_tlsext_host_name(ssl_stream_->native_handle(),
                                      const_cast<char*>(key_.host_.c_str()))) {
          throw boost::system::system_error{
              {static_cast<int>(::ERR_get_error()),
               boost::asio::error::get_ssl_category()}};
        }

        co_await boost::beast::get_lowest_layer(*ssl_stream_)
            .async_connect(results);
        co_await ssl_stream_->async_handshake(
            boost::asio::ssl::stream_base::client);
        std::cout << "[http] connect(" << key_.host_ << ":" << key_.port_
                  << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): ssl stream established" << std::endl;
      } else {
        std::cout << "[http] connect(" << key_.host_ << ":" << key_.port_
                  << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): creating tcp stream" << std::endl;
        stream_ = std::make_unique<boost::beast::tcp_stream>(executor);
        co_await stream_->async_connect(results);
        std::cout << "[http] connect(" << key_.host_ << ":" << key_.port_
                  << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): tcp stream established" << std::endl;
      }
    }

    boost::asio::awaitable<void> send_requests() {
      std::cout << "[http] send_requests(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;
      try {
        auto const send_request = [&](std::shared_ptr<request> request)
            -> boost::asio::awaitable<void> {
          std::cout << "[http] send_requests(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): got request: " << request->url_.encoded_target()
                    << std::endl;

          auto req =
              boost::beast::http::request<boost::beast::http::string_body>{
                  boost::beast::http::verb::get, request->url_.encoded_target(),
                  11};
          req.set(boost::beast::http::field::host, request->url_.host());
          req.set(boost::beast::http::field::user_agent,
                  BOOST_BEAST_VERSION_STRING);
          req.set(boost::beast::http::field::accept_encoding, "gzip");
          for (auto const& [k, v] : request->headers_) {
            req.set(k, v);
          }
          req.keep_alive(true);

          if (key_.ssl_) {
            co_await boost::beast::http::async_write(*ssl_stream_, req);
          } else {
            co_await boost::beast::http::async_write(*stream_, req);
          }
          ++n_sent_;
          std::cout << "[http] send_requests(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size() << "): request sent"
                    << std::endl;
        };

        if (!pending_requests_.empty()) {
          std::cout << "[http] send_requests(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): processing pending requests after reconnect"
                    << std::endl;
          for (auto const& request : pending_requests_) {
            co_await send_request(request);
          }
        }

        while (request_channel_.is_open()) {
          std::cout << "[http] send_requests(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): waiting for request" << std::endl;
          auto request = co_await request_channel_.async_receive();
          pending_requests_.push_back(request);
          co_await send_request(request);
        }
      } catch (std::exception const& ex) {
        std::cout << "[http] exception in send_requests: " << ex.what()
                  << std::endl;
        // throw ex;
      }
      std::cout << "[http] send_requests(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << "): END" << std::endl;
    }

    boost::asio::awaitable<void> receive_responses() {
      std::cout << "[http] receive_response(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;
      try {
        for (;;) {
          auto buffer = boost::beast::flat_buffer{};
          auto res =
              boost::beast::http::response<boost::beast::http::dynamic_body>{};

          std::cout << "[http] receive_response(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): waiting for response" << std::endl;
          if (key_.ssl_) {
            co_await boost::beast::http::async_read(*ssl_stream_, buffer, res);
          } else {
            co_await boost::beast::http::async_read(*stream_, buffer, res);
          }
          ++n_received_;

          std::cout << "[http] receive_response(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): response received, connection header: "
                    << res[boost::beast::http::field::connection] << std::endl;

          utl::verify(!pending_requests_.empty(),
                      "received response without pending request");
          auto req = pending_requests_.front();
          pending_requests_.pop_front();

          std::cout << "[http] receive_response(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): finishing pending request" << std::endl;
          co_await req->response_channel_.async_send(
              boost::system::error_code{}, std::move(res));
          std::cout << "[http] receive_response(" << key_.host_ << ":"
                    << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                    << ", r=" << n_received_
                    << ", p=" << pending_requests_.size()
                    << "): pending request finished" << std::endl;
        }
      } catch (std::exception const& ex) {
        std::cout << "[http] exception in receive_responses: " << ex.what()
                  << std::endl;
        // throw ex;
      }
      // TODO: boost::system::system_error
      std::cout << "[http] receive_response(" << key_.host_ << ":" << key_.port_
                << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << "): END" << std::endl;
    }

    boost::asio::awaitable<void> run() {
      using namespace boost::asio::experimental::awaitable_operators;
      std::cout << "[http] handle_connection(" << key_.host_ << ":"
                << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << ")" << std::endl;
      auto const self = shared_from_this();
      do {
        std::cout << "[http] handle_connection(" << key_.host_ << ":"
                  << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): (re)connecting" << std::endl;
        co_await self->connect();
        std::cout << "[http] handle_connection(" << key_.host_ << ":"
                  << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): connected, running send_requests + receive_responses"
                  << std::endl;
        co_await (self->receive_responses() || self->send_requests());
        std::cout << "[http] handle_connection(" << key_.host_ << ":"
                  << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                  << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                  << "): send / receive done" << std::endl;
      } while (!pending_requests_.empty());
      std::cout << "[http] handle_connection(" << key_.host_ << ":"
                << key_.port_ << ", ssl=" << key_.ssl_ << ", s=" << n_sent_
                << ", r=" << n_received_ << ", p=" << pending_requests_.size()
                << "): END" << std::endl;
    }

    connection_key key_{};
    std::unique_ptr<boost::beast::tcp_stream> stream_;
    std::unique_ptr<boost::asio::ssl::stream<boost::beast::tcp_stream>>
        ssl_stream_;
    boost::asio::experimental::channel<void(boost::system::error_code,
                                            std::shared_ptr<request>)>
        request_channel_;
    std::deque<std::shared_ptr<request>> pending_requests_;
    boost::asio::ssl::context ssl_ctx_{
        boost::asio::ssl::context::tlsv12_client};

    unsigned n_sent_{};
    unsigned n_received_{};
  };

  http_client() { std::cout << "[http] client ctor" << std::endl; }

  ~http_client() {
    std::cout << "[http] client dtor" << std::endl;
    for (auto const& [key, conn] : connections_) {
      conn->close();
    }
    std::cout << "[http] client dtor END" << std::endl;
  }

  boost::asio::awaitable<void> timeout(
      std::chrono::steady_clock::duration duration) {
    boost::asio::steady_timer timer(co_await boost::asio::this_coro::executor);
    timer.expires_after(duration);
    co_await timer.async_wait();
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

      boost::asio::co_spawn(executor, conn->run(), boost::asio::detached);

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
};

}  // namespace motis
