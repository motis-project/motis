#include "motis/http_client.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

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

namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;
namespace ssl = asio::ssl;

namespace motis {

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

constexpr auto const kMotisUserAgent =
    "MOTIS/" MOTIS_VERSION " " BOOST_BEAST_VERSION_STRING;

std::string http_client::error_category_impl::message(int ev) const {
  switch (static_cast<error>(ev)) {
    case error::success: return "success";
    case error::too_many_redirects: return "too many redirects";
    case error::request_failed: return "request failed (max retries reached)";
  }
  std::unreachable();
}

boost::system::error_category const& http_client_error_category() {
  static http_client::error_category_impl instance;
  return instance;
}

boost::system::error_code make_error_code(http_client::error e) {
  return boost::system::error_code(static_cast<int>(e),
                                   http_client_error_category());
}

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
  asio::experimental::channel<void(boost::system::error_code,
                                   http::response<http::dynamic_body>)>
      response_channel_;
  unsigned n_redirects_{};
};

struct http_client::connection
    : public std::enable_shared_from_this<connection> {
  template <typename Executor>
  connection(Executor const& executor,
             hash_map<connection_key, std::shared_ptr<connection>>& connections,
             connection_key key,
             std::chrono::seconds const timeout,
             proxy_settings const& proxy,
             std::size_t const max_in_flight = 1)
      : key_{std::move(key)},
        connections_{connections},
        unlimited_pipelining_{max_in_flight == kUnlimitedHttpPipelining},
        request_channel_{executor},
        requests_in_flight_{std::make_unique<
            asio::experimental::channel<void(boost::system::error_code)>>(
            executor, max_in_flight)},
        timeout_{timeout},
        proxy_{proxy} {
    ssl_ctx_.set_default_verify_paths();
    ssl_ctx_.set_verify_mode(ssl::verify_none);
    ssl_ctx_.set_options(ssl::context::default_workarounds |
                         ssl::context::no_sslv2 | ssl::context::no_sslv3 |
                         ssl::context::single_dh_use);
  }

  void close() {
    if (ssl_stream_) {
      auto ec = beast::error_code{};
      beast::get_lowest_layer(*ssl_stream_)
          .socket()
          .shutdown(asio::ip::tcp::socket::shutdown_both, ec);
      beast::get_lowest_layer(*ssl_stream_).socket().close(ec);
    } else if (stream_) {
      auto ec = beast::error_code{};
      beast::get_lowest_layer(*stream_).socket().shutdown(
          asio::ip::tcp::socket::shutdown_both, ec);
      beast::get_lowest_layer(*stream_).socket().close(ec);
    }
  }

  asio::awaitable<void> fail_all_requests(boost::system::error_code const ec) {
    while (!pending_requests_.empty()) {
      auto req = pending_requests_.front();
      pending_requests_.pop_front();
      co_await req->response_channel_.async_send(ec, {});
    }
  }

  asio::awaitable<void> connect() {
    auto executor = co_await asio::this_coro::executor;
    auto resolver = asio::ip::tcp::resolver{executor};

    auto const host = proxy_ ? proxy_.host_ : key_.host_;
    auto const port = proxy_ ? proxy_.port_ : key_.port_;

    auto const results = co_await resolver.async_resolve(host, port);
    ++n_connects_;
    if (ssl()) {
      ssl_stream_ =
          std::make_unique<ssl::stream<beast::tcp_stream>>(executor, ssl_ctx_);

      if (!SSL_set_tlsext_host_name(ssl_stream_->native_handle(),
                                    const_cast<char*>(host.c_str()))) {
        throw boost::system::system_error{{static_cast<int>(::ERR_get_error()),
                                           asio::error::get_ssl_category()}};
      }

      beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
      co_await beast::get_lowest_layer(*ssl_stream_).async_connect(results);
      co_await ssl_stream_->async_handshake(ssl::stream_base::client);
    } else {
      stream_ = std::make_unique<beast::tcp_stream>(executor);
      stream_->expires_after(timeout_);
      co_await stream_->async_connect(results);
    }

    requests_in_flight_->reset();
  }

  asio::awaitable<void> send_requests() {
    try {
      auto const send_request =
          [&](std::shared_ptr<request> request) -> asio::awaitable<void> {
        auto req = http::request<http::string_body>{
            http::verb::get, request->url_.encoded_target(), 11};
        req.set(http::field::host, request->url_.host());
        req.set(http::field::user_agent, kMotisUserAgent);
        req.set(http::field::accept_encoding, "gzip");
        for (auto const& [k, v] : request->headers_) {
          req.set(k, v);
        }
        req.keep_alive(true);

        if (!unlimited_pipelining_) {
          co_await requests_in_flight_->async_send(boost::system::error_code{});
        }

        if (ssl()) {
          beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
          co_await http::async_write(*ssl_stream_, req);
        } else {
          stream_->expires_after(timeout_);
          co_await http::async_write(*stream_, req);
        }
        ++n_sent_;
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
    } catch (std::exception const&) {
    }
  }

  asio::awaitable<void> receive_responses() {
    try {
      for (;;) {
        auto buffer = beast::flat_buffer{};
        auto res = http::response<http::dynamic_body>{};

        if (ssl()) {
          beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
          co_await http::async_read(*ssl_stream_, buffer, res);
        } else {
          stream_->expires_after(timeout_);
          co_await http::async_read(*stream_, buffer, res);
        }
        ++n_received_;

        if (!unlimited_pipelining_) {
          requests_in_flight_->try_receive([](auto const&) {});
        }

        utl::verify(!pending_requests_.empty(),
                    "received response without pending request");
        auto req = pending_requests_.front();
        pending_requests_.pop_front();
        n_current_retries_ = 0;

        auto const code = res.result_int();
        if (code >= 300 && code < 400) {  // redirect
          auto const location = res[http::field::location];
          auto next_url = boost::urls::url{};
          auto const resolve_result = boost::urls::resolve(
              req->url_, boost::urls::url{location}, next_url);
          if (resolve_result.has_error()) {
            co_await req->response_channel_.async_send(resolve_result.error(),
                                                       std::move(res));
            continue;
          }

          ++req->n_redirects_;
          req->url_ = next_url;
          if (req->n_redirects_ > 3) {
            co_await req->response_channel_.async_send(
                error::too_many_redirects, std::move(res));
          } else {
            co_await request_channel_.async_send(boost::system::error_code{},
                                                 req);
          }
          continue;
        }

        co_await req->response_channel_.async_send(boost::system::error_code{},
                                                   std::move(res));
      }
    } catch (std::exception const&) {
    }
  }

  asio::awaitable<void> run() {
    using namespace boost::asio::experimental::awaitable_operators;
    auto const self = shared_from_this();
    do {
      auto err = boost::system::error_code{error::request_failed};
      try {
        co_await self->connect();
        co_await (self->receive_responses() || self->send_requests());
      } catch (boost::system::system_error const& e) {
        err = e.code();
      }
      close();

      // if we get disconnected, don't use pipelining again
      unlimited_pipelining_ = false;
      auto executor = requests_in_flight_->get_executor();
      requests_in_flight_ = std::make_unique<
          asio::experimental::channel<void(boost::system::error_code)>>(
          executor, 1);

      // check if we have any more requests in the request channel and
      // receive the next one
      if (pending_requests_.empty() && request_channel_.is_open()) {
        auto request = co_await request_channel_.async_receive();
        pending_requests_.push_back(request);
      }

      if (!pending_requests_.empty()) {
        ++n_current_retries_;
        if (n_current_retries_ >= 3) {
          // fail all remaining pending requests
          co_await fail_all_requests(err);
        }
      }
    } while (!pending_requests_.empty());
    connections_.erase(key_);
  }

  bool ssl() const { return proxy_ ? proxy_.ssl_ : key_.ssl_; }

  connection_key key_{};
  hash_map<connection_key, std::shared_ptr<connection>>& connections_;
  bool unlimited_pipelining_{false};

  std::unique_ptr<beast::tcp_stream> stream_;
  std::unique_ptr<ssl::stream<beast::tcp_stream>> ssl_stream_;

  // the connection accepts new requests through the request_channel_
  asio::experimental::channel<void(boost::system::error_code,
                                   std::shared_ptr<request>)>
      request_channel_;
  // unless unlimited_pipelining_ is true, the requests_in_flight_
  // channel limits the number of requests that are in-flight (i.e. request
  // sent and waiting for response)
  std::unique_ptr<asio::experimental::channel<void(boost::system::error_code)>>
      requests_in_flight_;

  // requests that are sent and waiting for a response
  std::deque<std::shared_ptr<request>> pending_requests_;

  ssl::context ssl_ctx_{ssl::context::tlsv12_client};
  std::chrono::seconds timeout_;
  http_client::proxy_settings proxy_;

  unsigned n_sent_{};
  unsigned n_received_{};
  unsigned n_connects_{};
  // number of retries for the current request (reset after successful request)
  unsigned n_current_retries_{};
};

http_client::~http_client() {
  for (auto const& [key, conn] : connections_) {
    conn->close();
  }
}

asio::awaitable<http_response> http_client::get(
    boost::urls::url url, std::map<std::string, std::string> headers) {
  auto const https = url.scheme_id() == boost::urls::scheme::https;
  auto const key = connection_key{
      url.host(), url.has_port() ? url.port() : (https ? "443" : "80"), https};

  auto executor = co_await asio::this_coro::executor;
  if (auto const it = connections_.find(key); it == connections_.end()) {
    auto conn = std::make_shared<connection>(executor, connections_, key,
                                             timeout_, proxy_, 1);
    connections_[key] = conn;
    asio::co_spawn(executor, conn->run(), asio::detached);
  }

  auto req =
      std::make_shared<request>(std::move(url), std::move(headers), executor);
  co_await connections_[key]->request_channel_.async_send(
      boost::system::error_code{}, req);
  auto response = co_await req->response_channel_.async_receive();
  co_return response;
}

void http_client::set_proxy(boost::urls::url const& url) {
  proxy_.ssl_ = url.scheme_id() == boost::urls::scheme::https;
  proxy_.host_ = url.host();
  proxy_.port_ = url.has_port() ? url.port() : (proxy_.ssl_ ? "443" : "80");
}

}  // namespace motis
