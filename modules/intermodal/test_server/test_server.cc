#include "test_server.h"

#include "boost/asio/ip/tcp.hpp"
#include "boost/asio/strand.hpp"
#include "boost/beast/core/bind_handler.hpp"

#include "net/web_server/detect_session.h"
#include "net/web_server/fail.h"
#include "net/web_server/web_server_settings.h"

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

#if defined(NET_TLS)
namespace ssl = asio::ssl;
#endif

namespace net {

struct test_server::impl {
#if defined(NET_TLS)
  impl(asio::io_context& ioc, asio::ssl::context& ctx)
      : ioc_{ioc}, acceptor_{ioc}, ctx_{ctx} {}
#else
  explicit impl(asio::io_context& ioc) : ioc_{ioc}, acceptor_{ioc} {}
#endif

  void on_http_request(http_req_cb_t cb) const {
    settings_->http_req_cb_ = std::move(cb);
  }
  void on_ws_msg(ws_msg_cb_t cb) const {
    settings_->ws_msg_cb_ = std::move(cb);
  }
  void on_ws_open(ws_open_cb_t cb) const {
    settings_->ws_open_cb_ = std::move(cb);
  }
  void on_ws_close(ws_close_cb_t cb) const {
    settings_->ws_close_cb_ = std::move(cb);
  }
  void on_ws_upgrade_ok(ws_upgrade_ok_cb_t cb) const {
    settings_->ws_upgrade_ok_ = std::move(cb);
  }

  void init(std::string const& host, std::string const& port,
            boost::system::error_code& ec) {
    asio::ip::tcp::resolver resolver{ioc_};
    asio::ip::tcp::endpoint endpoint = *resolver.resolve({host, port});

    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
      fail(ec, "open");
      return;
    }

    acceptor_.set_option(asio::socket_base::reuse_address(true), ec);
    if (ec) {
      fail(ec, "set_option");
      return;
    }

    acceptor_.bind(endpoint, ec);
    if (ec) {
      fail(ec, "bind");
      return;
    }

    acceptor_.listen(asio::socket_base::max_listen_connections, ec);
    if (ec) {
      fail(ec, "listen");
      return;
    }
  }

  void run() {
    if (acceptor_.is_open()) {
      do_accept();
    }
  }

  void stop() { acceptor_.close(); }

  void set_timeout(std::chrono::seconds const& timeout) const {
    settings_->timeout_ = timeout;
  }

  void set_request_body_limit(std::uint64_t limit) const {
    settings_->request_body_limit_ = limit;
  }

  void set_request_queue_limit(std::size_t limit) const {
    settings_->request_queue_limit_ = limit;
  }

  void do_accept() {
    acceptor_.async_accept(
        asio::make_strand(ioc_),
        boost::beast::bind_front_handler(&impl::on_accept, this));
  }

  void on_accept(boost::system::error_code ec, asio::ip::tcp::socket socket) {
    if (!acceptor_.is_open()) {
      return;
    }

    if (ec) {
      fail(ec, "main accept");
    } else {
#if defined(NET_TLS)
      make_detect_session(std::move(socket), ctx_, settings_);
#else
      make_detect_session(std::move(socket), settings_);
#endif
    }
    do_accept();
  }

  asio::io_context& ioc_;
  tcp::acceptor acceptor_;
  web_server_settings_ptr settings_{std::make_shared<web_server_settings>()};

#if defined(NET_TLS)
  ssl::context& ctx_;
#endif
};

#if defined(NET_TLS)
test_server::test_server(asio::io_context& ioc, asio::ssl::context& ctx)
    : impl_{std::make_unique<impl>(ioc, ctx)} {}
#else
test_server::test_server(asio::io_context& ioc)
    : impl_{std::make_unique<impl>(ioc)} {}
#endif

test_server::~test_server() = default;

void test_server::init(std::string const& host, std::string const& port,
                       boost::system::error_code& ec) const {
  impl_->init(host, port, ec);
}

void test_server::run() const { impl_->run(); }

void test_server::stop() const { impl_->stop(); }

void test_server::set_timeout(std::chrono::seconds const& timeout) const {
  impl_->set_timeout(timeout);
}

void test_server::set_request_body_limit(std::uint64_t limit) const {
  impl_->set_request_body_limit(limit);
}

void test_server::set_request_queue_limit(std::size_t limit) const {
  impl_->set_request_queue_limit(limit);
}

void test_server::on_http_request(http_req_cb_t cb) const {
  impl_->on_http_request(std::move(cb));
}

void test_server::on_ws_msg(ws_msg_cb_t cb) const {
  impl_->on_ws_msg(std::move(cb));
}

void test_server::on_ws_open(ws_open_cb_t cb) const {
  impl_->on_ws_open(std::move(cb));
}

void test_server::on_ws_close(ws_close_cb_t cb) const {
  impl_->on_ws_close(std::move(cb));
}

void test_server::on_upgrade_ok(ws_upgrade_ok_cb_t cb) const {
  impl_->on_ws_upgrade_ok(std::move(cb));
}

}  // namespace net