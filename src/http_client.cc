#include "motis/http_client.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include <iostream>

#include "boost/asio/awaitable.hpp"
#include "boost/beast/http/dynamic_body.hpp"
#include "boost/beast/http/message.hpp"
#include "boost/url/url.hpp"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
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

namespace {
// TODO
boost::asio::awaitable<void> timeout(
    std::chrono::steady_clock::duration const duration) {
  boost::asio::steady_timer timer(co_await boost::asio::this_coro::executor);
  timer.expires_after(duration);
  co_await timer.async_wait();
}
}  // namespace

struct http_client::request {
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

struct http_client::connection
    : public std::enable_shared_from_this<connection> {
  template <typename Executor>
  connection(Executor const& executor,
             connection_key key,
             std::size_t const max_in_flight = 1)
      : key_{std::move(key)},
        unlimited_pipelining_{max_in_flight == kUnlimitedHttpPipelining},
        request_channel_{executor},
        requests_in_flight_{executor, max_in_flight} {
    ssl_ctx_.set_default_verify_paths();
    ssl_ctx_.set_verify_mode(boost::asio::ssl::verify_none);
    ssl_ctx_.set_options(boost::asio::ssl::context::default_workarounds |
                         boost::asio::ssl::context::no_sslv2 |
                         boost::asio::ssl::context::no_sslv3 |
                         boost::asio::ssl::context::single_dh_use);
  }

  void close() const {
    std::cout << "[http] closing connection to " << key_.host_ << ":"
              << key_.port_ << ": sent " << n_sent_ << " requests, received "
              << n_received_ << " responses, " << pending_requests_.size()
              << " still pending, " << n_connects_ << " connects" << std::endl;
    if (ssl()) {
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
    auto executor = co_await boost::asio::this_coro::executor;
    auto resolver = boost::asio::ip::tcp::resolver{executor};

    auto const results =
        co_await resolver.async_resolve(key_.host_, key_.port_);
    ++n_connects_;
    if (ssl()) {
      ssl_stream_ =
          std::make_unique<boost::asio::ssl::stream<boost::beast::tcp_stream>>(
              executor, ssl_ctx_);

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
    } else {
      stream_ = std::make_unique<boost::beast::tcp_stream>(executor);
      co_await stream_->async_connect(results);
    }

    requests_in_flight_.reset();
  }

  boost::asio::awaitable<void> send_requests() {
    try {
      auto const send_request = [&](std::shared_ptr<request> request)
          -> boost::asio::awaitable<void> {
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

        if (!unlimited_pipelining_) {
          co_await requests_in_flight_.async_send(boost::system::error_code{});
        }

        if (ssl()) {
          co_await boost::beast::http::async_write(*ssl_stream_, req);
        } else {
          co_await boost::beast::http::async_write(*stream_, req);
        }
        ++n_sent_;
        std::cout << "[http::send_requests] sent " << n_sent_ << ", received "
                  << n_received_ << ", " << pending_requests_.size()
                  << " pending" << std::endl;
      };

      if (!pending_requests_.empty()) {
        // after reconnect, send pending requests again
        for (auto const& request : pending_requests_) {
          co_await send_request(request);
        }
      }

      while (request_channel_.is_open()) {
        auto request = co_await request_channel_.async_receive();
        pending_requests_.push_back(request);
        co_await send_request(request);
      }
    } catch (std::exception const& ex) {
      std::cout << "[http] exception in send_requests: " << ex.what()
                << std::endl;
      // throw ex;
    }
  }

  boost::asio::awaitable<void> receive_responses() {
    try {
      for (;;) {
        auto buffer = boost::beast::flat_buffer{};
        auto res =
            boost::beast::http::response<boost::beast::http::dynamic_body>{};

        if (ssl()) {
          co_await boost::beast::http::async_read(*ssl_stream_, buffer, res);
        } else {
          co_await boost::beast::http::async_read(*stream_, buffer, res);
        }
        ++n_received_;

        if (!unlimited_pipelining_) {
          requests_in_flight_.try_receive([](auto const&) {});
        }

        utl::verify(!pending_requests_.empty(),
                    "received response without pending request");
        auto req = pending_requests_.front();
        pending_requests_.pop_front();

        co_await req->response_channel_.async_send(boost::system::error_code{},
                                                   std::move(res));
      }
    } catch (std::exception const& ex) {
      std::cout << "[http] exception in receive_responses: " << ex.what()
                << std::endl;
      // throw ex;
    }
    // TODO: boost::system::system_error
  }

  boost::asio::awaitable<void> run() {
    using namespace boost::asio::experimental::awaitable_operators;
    auto const self = shared_from_this();
    do {
      co_await self->connect();
      co_await (self->receive_responses() || self->send_requests());
      close();
    } while (!pending_requests_.empty());
  }

  bool ssl() const { return key_.ssl_; }

  connection_key key_{};
  bool unlimited_pipelining_{false};

  std::unique_ptr<boost::beast::tcp_stream> stream_;
  std::unique_ptr<boost::asio::ssl::stream<boost::beast::tcp_stream>>
      ssl_stream_;

  // the connection accepts new requests through the request_channel_
  boost::asio::experimental::channel<void(boost::system::error_code,
                                          std::shared_ptr<request>)>
      request_channel_;
  // unless unlimited_pipelining_ is true, the requests_in_flight_
  // channel limits the number of requests that are in-flight (i.e. request
  // sent and waiting for response)
  boost::asio::experimental::channel<void(boost::system::error_code)>
      requests_in_flight_;

  // requests that are sent and waiting for a response
  std::deque<std::shared_ptr<request>> pending_requests_;

  boost::asio::ssl::context ssl_ctx_{boost::asio::ssl::context::tlsv12_client};

  unsigned n_sent_{};
  unsigned n_received_{};
  unsigned n_connects_{};
};

http_client::~http_client() {
  for (auto const& [key, conn] : connections_) {
    conn->close();
  }
}

boost::asio::awaitable<http_response> http_client::get(
    boost::urls::url url, std::map<std::string, std::string> headers) {
  auto const https = url.scheme_id() == boost::urls::scheme::https;
  auto const key = connection_key{
      url.host(), url.has_port() ? url.port() : (https ? "443" : "80"), https};

  std::cout << "[http] get(" << key.host_ << ":" << key.port_
            << ", ssl=" << key.ssl_ << "), target=" << url.encoded_target()
            << std::endl;

  auto executor = co_await boost::asio::this_coro::executor;
  if (auto const it = connections_.find(key); it == connections_.end()) {
    auto conn = std::make_shared<connection>(executor, key);
    connections_[key] = conn;
    boost::asio::co_spawn(executor, conn->run(), boost::asio::detached);
  }

  auto req =
      std::make_shared<request>(std::move(url), std::move(headers), executor);
  co_await connections_[key]->request_channel_.async_send(
      boost::system::error_code{}, req);
  auto response = co_await req->response_channel_.async_receive();
  co_return response;
}

}  // namespace motis
