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

std::string get_http_body(http_response const& res) {
  std::cout << "[http_client] get_http_body called, content-encoding="
            << res[http::field::content_encoding] << std::endl;
  auto body = beast::buffers_to_string(res.body().data());
  if (res[http::field::content_encoding] == "gzip") {
    std::cout << "[http_client] decompressing gzip encoded body" << std::endl;
    auto const src = boost::iostreams::array_source{body.data(), body.size()};
    auto is = boost::iostreams::filtering_istream{};
    auto os = std::stringstream{};
    is.push(boost::iostreams::gzip_decompressor{});
    is.push(src);
    boost::iostreams::copy(is, os);
    body = os.str();
  }
  std::cout << "[http_client] get_http_body finished, size=" << body.size()
            << std::endl;
  return body;
}

std::string http_client::error_category_impl::message(int ev) const {
  std::cout << "[http_client] error_category_impl::message called, ev=" << ev
            << std::endl;
  switch (static_cast<error>(ev)) {
    case error::success: return "success";
    case error::too_many_redirects: return "too many redirects";
    case error::request_failed: return "request failed (max retries reached)";
  }
  std::unreachable();
}

boost::system::error_category const& http_client_error_category() {
  std::cout << "[http_client] http_client_error_category requested"
            << std::endl;
  static http_client::error_category_impl instance;
  return instance;
}

boost::system::error_code make_error_code(http_client::error e) {
  std::cout << "[http_client] make_error_code called with value="
            << static_cast<int>(e) << std::endl;
  return boost::system::error_code(static_cast<int>(e),
                                   http_client_error_category());
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
        response_channel_{executor} {
    std::cout << "[http_client] request constructed url=" << url_.buffer()
              << " method=" << http::to_string(method_)
              << " headers=" << headers_.size() << " body_size=" << body_.size()
              << std::endl;
  }

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
    std::cout << "[http_client] connection constructed " << conn_id()
              << " ssl=" << std::boolalpha << key_.ssl_
              << " max_in_flight=" << max_in_flight << std::noboolalpha
              << std::endl;
    ssl_ctx_.set_default_verify_paths();
    ssl_ctx_.set_verify_mode(ssl::verify_none);
    ssl_ctx_.set_options(ssl::context::default_workarounds |
                         ssl::context::no_sslv2 | ssl::context::no_sslv3 |
                         ssl::context::single_dh_use);
  }

  void close() {
    std::cout << "[http_client] connection::close " << conn_id() << std::endl;
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
    std::cout << "[http_client] connection::fail_all_requests " << conn_id()
              << " pending=" << pending_requests_.size()
              << " ec=" << ec.message() << std::endl;
    while (!pending_requests_.empty()) {
      auto req = pending_requests_.front();
      pending_requests_.pop_front();
      std::cout << "[http_client] failing request " << conn_id()
                << " url=" << req->url_.buffer() << std::endl;
      co_await req->response_channel_.async_send(ec, {});
    }
  }

  asio::awaitable<void> connect() {
    auto const conn = conn_id();
    std::cout << "[http_client] connection::connect " << conn
              << " ssl=" << std::boolalpha << ssl() << std::noboolalpha
              << std::endl;
    auto executor = co_await asio::this_coro::executor;
    auto resolver = asio::ip::tcp::resolver{executor};

    auto const host = proxy_ ? proxy_.host_ : key_.host_;
    auto const port = proxy_ ? proxy_.port_ : key_.port_;

    std::cout << "[http_client] resolving host=" << host << " port=" << port
              << " for " << conn << std::endl;
    auto const results = co_await resolver.async_resolve(host, port);
    std::cout << "[http_client] resolved host=" << host << " port=" << port
              << " for " << conn << std::endl;
    ++n_connects_;
    if (ssl()) {
      std::cout << "[http_client] establishing SSL connection for " << conn
                << std::endl;
      ssl_stream_ =
          std::make_unique<ssl::stream<beast::tcp_stream>>(executor, ssl_ctx_);

      if (!SSL_set_tlsext_host_name(ssl_stream_->native_handle(),
                                    const_cast<char*>(host.c_str()))) {
        throw boost::system::system_error{{static_cast<int>(::ERR_get_error()),
                                           asio::error::get_ssl_category()}};
      }

      beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
      co_await beast::get_lowest_layer(*ssl_stream_).async_connect(results);
      std::cout << "[http_client] ssl connect complete, handshake starting for "
                << conn << std::endl;
      co_await ssl_stream_->async_handshake(ssl::stream_base::client);
    } else {
      std::cout << "[http_client] establishing TCP connection for " << conn
                << std::endl;
      stream_ = std::make_unique<beast::tcp_stream>(executor);
      stream_->expires_after(timeout_);
      co_await stream_->async_connect(results);
    }

    std::cout << "[http_client] connection::connect success " << conn
              << std::endl;
    requests_in_flight_->reset();
  }

  asio::awaitable<void> send_requests() {
    auto const conn = conn_id();
    std::cout << "[http_client] connection::send_requests start " << conn
              << std::endl;
    try {
      auto const send_request =
          [&](std::shared_ptr<request> request) -> asio::awaitable<void> {
        std::cout << "[http_client] preparing request " << conn
                  << " url=" << request->url_.buffer()
                  << " method=" << http::to_string(request->method_)
                  << std::endl;
        auto req = http::request<http::string_body>{
            request->method_, request->url_.encoded_target(), 11};
        req.set(http::field::host, request->url_.host());
        req.set(http::field::user_agent, kMotisUserAgent);
        req.set(http::field::accept_encoding, "gzip");
        for (auto const& [k, v] : request->headers_) {
          std::cout << "[http_client]   header(" << conn << "): " << k << "="
                    << v << std::endl;
          req.set(k, v);
        }
        req.keep_alive(true);

        if (request->method_ == http::verb::post) {
          std::cout << "[http_client] request has body " << conn
                    << " size=" << request->body_.size() << std::endl;
          req.body() = request->body_;
          req.prepare_payload();
        }

        if (!unlimited_pipelining_) {
          std::cout << "[http_client] waiting for requests_in_flight slot "
                    << conn << std::endl;
          co_await requests_in_flight_->async_send(boost::system::error_code{});
        }

        if (ssl()) {
          std::cout << "[http_client] writing request over SSL " << conn
                    << std::endl;
          beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
          co_await http::async_write(*ssl_stream_, req);
        } else {
          std::cout << "[http_client] writing request over TCP " << conn
                    << std::endl;
          stream_->expires_after(timeout_);
          co_await http::async_write(*stream_, req);
        }
        ++n_sent_;
        std::cout << "[http_client] request sent " << conn
                  << " total_sent=" << n_sent_ << std::endl;
      };

      if (!pending_requests_.empty()) {
        std::cout << "[http_client] resending " << pending_requests_.size()
                  << " pending requests for " << conn << std::endl;
        // after reconnect, send pending requests again
        for (auto const& request : pending_requests_) {
          co_await send_request(request);
        }
      }

      while (request_channel_.is_open()) {
        std::cout << "[http_client] waiting for new request from channel "
                  << conn << std::endl;
        auto request = co_await request_channel_.async_receive();
        pending_requests_.push_back(request);
        std::cout << "[http_client] received request " << conn
                  << " pending=" << pending_requests_.size() << std::endl;
        co_await send_request(request);
      }
      std::cout << "[http_client] request_channel closed " << conn << std::endl;
    } catch (std::exception const& e) {
      std::cout << "[http_client] send_requests exception " << conn << ": "
                << e.what() << std::endl;
    }
    std::cout << "[http_client] connection::send_requests exit " << conn
              << std::endl;
  }
  asio::awaitable<void> receive_responses() {
    auto const conn = conn_id();
    std::cout << "[http_client] connection::receive_responses start " << conn
              << std::endl;
    try {
      for (;;) {
        auto buffer = beast::flat_buffer{};
        auto res = http::response<http::dynamic_body>{};

        if (ssl()) {
          std::cout << "[http_client] waiting for HTTPS response from " << conn
                    << " timeout=" << timeout_ << std::endl;
          beast::get_lowest_layer(*ssl_stream_).expires_after(timeout_);
          co_await http::async_read(*ssl_stream_, buffer, res);
        } else {
          std::cout << "[http_client] waiting for HTTP response from " << conn
                    << " timeout=" << timeout_ << std::endl;
          stream_->expires_after(timeout_);
          co_await http::async_read(*stream_, buffer, res);
        }
        ++n_received_;
        std::cout << "[http_client] response received " << conn
                  << " status=" << res.result_int()
                  << " total_received=" << n_received_ << std::endl;

        if (!unlimited_pipelining_) {
          std::cout << "[http_client] releasing requests_in_flight slot "
                    << conn << std::endl;
          requests_in_flight_->try_receive([](auto const&) {});
        }

        utl::verify(!pending_requests_.empty(),
                    "received response without pending request");
        auto req = pending_requests_.front();
        pending_requests_.pop_front();
        std::cout << "[http_client] matching response to url="
                  << req->url_.buffer() << " for " << conn << std::endl;
        n_current_retries_ = 0;

        std::cout << "[http_client] delivering response to caller " << conn
                  << std::endl;
        co_await req->response_channel_.async_send(boost::system::error_code{},
                                                   std::move(res));
      }
    } catch (std::exception const& e) {
      std::cout << "[http_client] receive_responses exception " << conn << ": "
                << e.what() << std::endl;
    }
    std::cout << "[http_client] connection::receive_responses exit " << conn
              << std::endl;
  }

  asio::awaitable<void> run() {
    auto const conn = conn_id();
    std::cout << "[http_client] connection::run start " << conn << std::endl;
    using namespace boost::asio::experimental::awaitable_operators;
    auto const self = shared_from_this();
    do {
      auto err = boost::system::error_code{error::request_failed};
      try {
        co_await self->connect();
        co_await (self->receive_responses() || self->send_requests());
      } catch (boost::system::system_error const& e) {
        std::cout << "[http_client] connection::run caught system_error "
                  << conn << ": " << e.code().message() << std::endl;
        err = e.code();
      }
      close();
      std::cout << "[http_client] connection::run closed connection " << conn
                << std::endl;

      // if we get disconnected, don't use pipelining again
      unlimited_pipelining_ = false;
      std::cout << "[http_client] resetting requests_in_flight channel " << conn
                << std::endl;
      auto executor = requests_in_flight_->get_executor();
      requests_in_flight_ = std::make_unique<
          asio::experimental::channel<void(boost::system::error_code)>>(
          executor, 1);

      // check if we have any more requests in the request channel and
      // receive the next one
      if (pending_requests_.empty() && request_channel_.is_open()) {
        std::cout << "[http_client] waiting for next request after reconnect "
                  << conn << std::endl;
        auto request = co_await request_channel_.async_receive();
        pending_requests_.push_back(request);
        std::cout << "[http_client] received request after reconnect " << conn
                  << " pending=" << pending_requests_.size() << std::endl;
      }

      if (!pending_requests_.empty()) {
        ++n_current_retries_;
        std::cout << "[http_client] retry count for current request " << conn
                  << "=" << n_current_retries_ << std::endl;
        if (n_current_retries_ >= 3) {
          // fail all remaining pending requests
          std::cout << "[http_client] retry limit reached, failing all pending "
                       "requests "
                    << conn << std::endl;
          co_await fail_all_requests(err);
        }
      }
    } while (!pending_requests_.empty());
    std::cout << "[http_client] connection::run finished, erasing connection "
              << conn << std::endl;
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

  std::string conn_id() const { return key_.host_ + ":" + key_.port_; }
};

http_client::~http_client() {
  std::cout << "[http_client] destructor called, connections="
            << connections_.size() << std::endl;
  for (auto const& [key, conn] : connections_) {
    std::cout << "[http_client] closing connection " << key.host_ << ":"
              << key.port_ << std::endl;
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
  std::cout << "[http_client] connection key " << conn
            << " ssl=" << std::boolalpha << key.ssl_ << std::noboolalpha
            << std::endl;

  auto executor = co_await asio::this_coro::executor;
  if (auto const it = connections_.find(key); it == connections_.end()) {
    std::cout << "[http_client] creating new connection for " << conn
              << std::endl;
    auto new_conn = std::make_shared<connection>(executor, connections_, key,
                                                 timeout_, proxy_, 1);
    connections_[key] = new_conn;
    asio::co_spawn(executor, new_conn->run(), asio::detached);
  } else {
    std::cout << "[http_client] reusing existing connection for " << conn
              << std::endl;
  }

  std::cout << "[http_client] sending request through connection " << conn
            << std::endl;
  co_await connections_[key]->request_channel_.async_send(
      boost::system::error_code{}, r);
  std::cout << "[http_client] waiting for response " << conn << std::endl;
  auto ec = boost::system::error_code{};
  auto response = co_await r->response_channel_.async_receive(
      asio::redirect_error(asio::use_awaitable, ec));
  if (ec) {
    std::cout << "[http_client] request error on " << conn << ": "
              << ec.message() << std::endl;
    throw boost::system::system_error{ec};
  }
  std::cout << "[http_client] response received " << conn
            << " status=" << response.result_int() << std::endl;
  co_return std::move(response);
}

asio::awaitable<http_response> http_client::get(
    boost::urls::url url, std::map<std::string, std::string> headers) {
  std::cout << "[http_client] get called url=" << url.buffer()
            << " headers=" << headers.size() << std::endl;
  auto executor = co_await asio::this_coro::executor;
  co_return co_await req(std::make_shared<request>(
      std::move(url), http::verb::get, std::move(headers), "", executor));
}

asio::awaitable<http_response> http_client::post(
    boost::urls::url url,
    std::map<std::string, std::string> headers,
    std::string body) {
  std::cout << "[http_client] post called url=" << url.buffer()
            << " headers=" << headers.size() << " body_size=" << body.size()
            << std::endl;
  auto executor = co_await asio::this_coro::executor;
  co_return co_await req(
      std::make_shared<request>(std::move(url), http::verb::post,
                                std::move(headers), std::move(body), executor));
}

asio::awaitable<http_response> http_client::req(
    std::shared_ptr<request> const r) {
  std::cout << "[http_client] req called url=" << r->url_.buffer() << std::endl;
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
      std::cout << "[http_client] redirect status without location for "
                << current_request->url_.buffer() << std::endl;
      co_return response;
    }

    ++redirects;
    if (redirects > 3) {
      std::cout << "[http_client] redirect limit exceeded for "
                << current_request->url_.buffer() << std::endl;
      throw boost::system::system_error{
          make_error_code(error::too_many_redirects)};
    }

    auto next_url = boost::urls::url{};
    auto const resolve_result = boost::urls::resolve(
        current_request->url_, boost::urls::url{location}, next_url);
    if (resolve_result.has_error()) {
      std::cout << "[http_client] redirect resolve error for "
                << current_request->url_.buffer() << ": "
                << resolve_result.error().message() << std::endl;
      throw boost::system::system_error{resolve_result.error()};
    }

    std::cout << "[http_client] following redirect "
              << current_request->url_.buffer() << " -> " << next_url.buffer()
              << std::endl;

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
  std::cout << "[http_client] set_proxy url=" << url.buffer() << std::endl;
  proxy_.ssl_ = url.scheme_id() == boost::urls::scheme::https;
  proxy_.host_ = url.host();
  proxy_.port_ = url.has_port() ? url.port() : (proxy_.ssl_ ? "443" : "80");
  std::cout << "[http_client] proxy configured host=" << proxy_.host_
            << " port=" << proxy_.port_ << " ssl=" << std::boolalpha
            << proxy_.ssl_ << std::noboolalpha << std::endl;
}

}  // namespace motis
