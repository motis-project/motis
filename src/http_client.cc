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

#include "boost/asio/awaitable.hpp"
#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/awaitable_operators.hpp"
#include "boost/asio/experimental/channel.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/ssl.hpp"

#include "boost/beast/core.hpp"
#include "boost/beast/http.hpp"
#include "boost/beast/ssl/ssl_stream.hpp"
#include "boost/beast/version.hpp"

#include "boost/iostreams/copy.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "boost/iostreams/filtering_streambuf.hpp"

#include "utl/verify.h"

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

/*std::string get_http_body(http_response const& res) {
auto body = beast::buffers_to_string(res.body().data());
if (res[http::field::content_encoding] == "gzip") {
auto const src = boost::iostreams::array_source{body.data(), body.size()};
auto is = boost::iostreams::filtering_istream{};
auto os = std::stringstream{};
is.push(boost::iostreams::gzip_decompressor{});
is.push(src);
boost::iostreams::copy(is, os);
body = os.str();
}
return body;
}*/

std::string http_client::error_category_impl::message(int const ev) const {
  switch (static_cast<error>(ev)) {
    case error::success: return "success";
    case error::too_many_redirects: return "too many redirects";
    case error::request_failed: return "request failed (max retries reached)";
    case error::timeout: return "request timeout reached";
  }
  std::unreachable();
}

boost::system::error_category const& http_client_error_category() {
  static http_client::error_category_impl instance;
  return instance;
}

boost::system::error_code make_error_code(http_client::error const e) {
  return boost::system::error_code{static_cast<int>(e),
                                   http_client_error_category()};
}

struct http_client::request {
  template <typename Executor>
  request(boost::urls::url&& url,
          http::verb const method,
          std::map<std::string, std::string>&& headers,
          std::string&& body,
          Executor const& executor)
      : url_{std::move(url)},
        method_{method},
        headers_{std::move(headers)},
        body_{std::move(body)},
        response_channel_{executor} {}

  boost::urls::url url_{};
  http::verb method_{http::verb::get};
  std::map<std::string, std::string> headers_{};
  std::string body_{};
  asio::experimental::channel<void(boost::system::error_code,
                                   http::response<http::dynamic_body>)>
      response_channel_;
  unsigned n_redirects_{};
};

struct http_client::connection
    : public std::enable_shared_from_this<connection> {
  template <typename Executor>
  connection(Executor const& executor,
             std::weak_ptr<http_client> client,
             connection_key key,
             std::chrono::seconds const timeout,
             proxy_settings const& proxy,
             std::size_t const max_in_flight = 1)
      : key_{std::move(key)},
        client_{client},
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

    auto const& host = proxy_ ? proxy_.host_ : key_.host_;
    auto const& port = proxy_ ? proxy_.port_ : key_.port_;

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

      beast::get_lowest_layer(*ssl_stream_)
          .expires_after(std::chrono::seconds{5});
      co_await beast::get_lowest_layer(*ssl_stream_).async_connect(results);
      co_await ssl_stream_->async_handshake(ssl::stream_base::client);
    } else {
      stream_ = std::make_unique<beast::tcp_stream>(executor);
      stream_->expires_after(std::chrono::seconds{5});
      co_await stream_->async_connect(results);
    }
    requests_in_flight_->reset();
  }

  asio::awaitable<void> send_requests() {
    try {
      auto const send_request =
          [&](std::shared_ptr<request> request) -> asio::awaitable<void> {
        auto req = http::request<http::string_body>{
            request->method_, request->url_.encoded_target(), 11};
        req.set(http::field::host, request->url_.host());
        req.set(http::field::user_agent, kMotisUserAgent);
        req.set(http::field::accept_encoding, "gzip");
        for (auto const& [k, v] : request->headers_) {
          req.set(k, v);
        }
        req.keep_alive(true);

        if (request->method_ == http::verb::post) {
          req.body() = request->body_;
          req.prepare_payload();
        }

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

        auto p = http::response_parser<http::dynamic_body>{};
        p.eager(true);
        p.body_limit(kBodySizeLimit);

        if (ssl()) {
          beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
          co_await http::async_read(*ssl_stream_, buffer, p);
        } else {
          stream_->expires_after(timeout_);
          co_await http::async_read(*stream_, buffer, p);
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

        co_await req->response_channel_.async_send(boost::system::error_code{},
                                                   p.release());
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

    auto const c = client_.lock();
    if (c) {
      c->connections_.erase(key_);
    }
  }

  bool ssl() const { return proxy_ ? proxy_.ssl_ : key_.ssl_; }

  connection_key key_{};
  std::weak_ptr<http_client> client_;
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

asio::awaitable<http_response> http_client::perform_request(
    std::shared_ptr<request> const r) {
  auto const https = r->url_.scheme_id() == boost::urls::scheme::https;
  auto const key = connection_key{
      r->url_.host(),
      r->url_.has_port() ? r->url_.port() : (https ? "443" : "80"), https};

  auto const conn = key.host_ + ":" + key.port_;

  auto executor = co_await asio::this_coro::executor;
  if (auto const it = connections_.find(key); it == connections_.end()) {
    auto new_conn = std::make_shared<connection>(executor, shared_from_this(),
                                                 key, timeout_, proxy_, 1);
    connections_[key] = new_conn;
    asio::co_spawn(executor, new_conn->run(), asio::detached);
  }

  co_await connections_[key]->request_channel_.async_send(
      boost::system::error_code{}, r);
  auto ec = boost::system::error_code{};
  auto response = co_await r->response_channel_.async_receive(
      asio::redirect_error(asio::use_awaitable, ec));
  if (ec) {
    throw boost::system::system_error{ec};
  }
  co_return std::move(response);
}

asio::awaitable<http_response> http_client::get(
    boost::urls::url url, std::map<std::string, std::string> headers) {
  auto executor = co_await asio::this_coro::executor;
  co_return co_await req(std::make_shared<request>(
      std::move(url), http::verb::get, std::move(headers), "", executor));
}

asio::awaitable<http_response> http_client::post(
    boost::urls::url url,
    std::map<std::string, std::string> headers,
    std::string body) {
  auto executor = co_await asio::this_coro::executor;
  co_return co_await req(
      std::make_shared<request>(std::move(url), http::verb::post,
                                std::move(headers), std::move(body), executor));
}

asio::awaitable<http_response> http_client::req(
    std::shared_ptr<request> const r) {
  auto executor = co_await asio::this_coro::executor;

  auto current_request = r;
  auto redirects = r->n_redirects_;

  while (true) {
    auto response = co_await perform_request(current_request);

    auto const status = response.result_int();
    if (status < 300 || status >= 400) {
      co_return response;
    }

    auto const location = response[http::field::location];
    if (location.empty()) {
      co_return response;
    }

    ++redirects;
    if (redirects > 3) {
      throw boost::system::system_error{
          make_error_code(error::too_many_redirects)};
    }

    auto next_url = boost::urls::url{};
    auto const resolve_result = boost::urls::resolve(
        current_request->url_, boost::urls::url{location}, next_url);
    if (resolve_result.has_error()) {
      throw boost::system::system_error{resolve_result.error()};
    }

    auto method = current_request->method_;
    auto headers = std::move(current_request->headers_);
    auto body = std::move(current_request->body_);
    current_request = std::make_shared<request>(std::move(next_url), method,
                                                std::move(headers),
                                                std::move(body), executor);
    current_request->n_redirects_ = redirects;
  }
}

void http_client::set_proxy(boost::urls::url const& url) {
  proxy_.ssl_ = url.scheme_id() == boost::urls::scheme::https;
  proxy_.host_ = url.host();
  proxy_.port_ = url.has_port() ? url.port() : (proxy_.ssl_ ? "443" : "80");
}

asio::awaitable<void> http_client::shutdown() {
  while (!connections_.empty()) {
    auto const it = connections_.begin();
    auto const con = it->second;
    connections_.erase(it);
    co_await con->fail_all_requests(make_error_code(error::timeout));
    con->unlimited_pipelining_ = true;
    con->requests_in_flight_->cancel();
    con->close();
    con->request_channel_.close();
  }
}

}  // namespace motis
