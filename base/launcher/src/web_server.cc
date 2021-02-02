#include "motis/launcher/web_server.h"

#include <functional>
#include <iostream>

#include "boost/beast/version.hpp"
#include "boost/filesystem.hpp"

#include "net/web_server/responses.h"
#include "net/web_server/serve_static.h"
#include "net/web_server/web_server.h"

#include "motis/core/common/logging.h"
#include "motis/launcher/load_server_certificate.h"

#if defined(NET_TLS)
namespace ssl = boost::asio::ssl;
#endif

namespace fs = boost::filesystem;
using namespace motis::module;

namespace motis::launcher {

struct web_server::impl {
#if defined(NET_TLS)
  impl(boost::asio::io_service& ios, receiver& recvr)
      : ctx_{ssl::context::tlsv12},
        ios_{ios},
        receiver_{recvr},
        server_{ios, ctx_} {}
#else
  impl(boost::asio::io_service& ios, receiver& recvr)
      : ios_{ios}, receiver_{recvr}, server_{ios} {}
#endif

  void listen(std::string const& host, std::string const& port,
#if defined(NET_TLS)
              std::string const& cert_path, std::string const& priv_key_path,
              std::string const& dh_path,
#endif
              std::string const& log_path, std::string const& static_path,
              boost::system::error_code& ec) {
#if defined(NET_TLS)
    load_server_certificate(ctx_, cert_path, priv_key_path, dh_path);
#endif

    server_.on_http_request([this](net::web_server::http_req_t const& req,
                                   net::web_server::http_res_cb_t const& cb,
                                   bool) { on_http_request(req, cb); });
    server_.on_ws_msg(
        [this](net::ws_session_ptr const& session, std::string const& msg,
               net::ws_msg_type type) { on_ws_msg(session, msg, type); });
    server_.set_timeout(std::chrono::seconds(120));
    server_.init(host, port, ec);
    log_path_ = log_path;
    if (!log_path_.empty()) {
      reset_logging(true);
    }
    try {
      if (!static_path.empty() && fs::is_directory(static_path)) {
        static_file_path_ = fs::canonical(static_path).string();
        serve_static_files_ = true;
      }
    } catch (fs::filesystem_error const& e) {
      std::cerr << "Static file directory not found: " << e.what() << std::endl;
    }
    if (serve_static_files_) {
      std::cout << "Serving static files from " << static_file_path_
                << std::endl;
    } else {
      std::cout << "Not serving static files" << std::endl;
    }
    if (!ec) {
      server_.run();
    }
  }

  void stop() { server_.stop(); }

  void on_http_request(net::web_server::http_req_t const& req,
                       net::web_server::http_res_cb_t const& cb) {
    using namespace boost::beast::http;

    auto const build_response = [req](msg_ptr const& response) {
      net::web_server::string_res_t res{
          response == nullptr ? status::ok
          : response->get()->content_type() == MsgContent_MotisError
              ? status::internal_server_error
              : status::ok,
          req.version()};
      res.set(field::access_control_allow_origin, "*");
      res.set(field::access_control_allow_headers,
              "X-Requested-With, Content-Type, Accept, Authorization");
      res.set(field::access_control_allow_methods, "GET, POST, OPTIONS");
      res.set(field::access_control_max_age, "3600");
      res.keep_alive(req.keep_alive());
      res.set(field::server, BOOST_BEAST_VERSION_STRING);

      if (response != nullptr &&
          response->get()->content_type() == MsgContent_HTTPResponse) {
        auto const http_res = motis_content(HTTPResponse, response);
        res.result(http_res->status() == HTTPStatus_OK
                       ? http_res->content()->size() != 0 ? status::ok
                                                          : status::no_content
                       : status::internal_server_error);

        res.body() = http_res->content()->str();
        for (auto const& h : *http_res->headers()) {
          res.set(h->name()->str(), h->value()->str());
        }
      } else {
        res.set(field::content_type, "application/json");
        res.body() = response == nullptr ? "" : response->to_json();
      }

      res.prepare_payload();
      return res;
    };

    std::string req_msg;
    switch (req.method()) {
      case verb::options: return cb(build_response(nullptr));
      case verb::post:
        req_msg = req.body();
        if (req_msg.empty()) {
          req_msg = make_no_msg(std::string{req.target()})->to_json();
        }
        break;
      case verb::head:
      case verb::get:
        if (serve_static_files_ &&
            net::serve_static_file(static_file_path_, req, cb)) {
          return;
        } else {
          req_msg = make_no_msg(std::string{req.target()})->to_json();
          break;
        }
      default:
        return cb(build_response(make_error_msg(
            std::make_error_code(std::errc::operation_not_supported))));
    }

    return on_req(req_msg, false,
                  [cb, build_response](msg_ptr const& response) {
                    cb(build_response(response));
                  });
  }

  void on_ws_msg(net::ws_session_ptr const& session, std::string const& msg,
                 net::ws_msg_type type) {
    bool const binary = type == net::ws_msg_type::BINARY;
    return on_req(msg, binary,
                  [session, type, binary](msg_ptr const& response) {
                    if (auto s = session.lock()) {
                      s->send(encode_msg(response, binary), type,
                              [](boost::system::error_code, size_t) {});
                    }
                  });
  }

  void on_req(std::string const& request, bool binary,
              std::function<void(msg_ptr const&)> const& cb) {
    msg_ptr err;
    int req_id = 0;
    try {
      auto const req = decode_msg(request, binary);
      log_request(req);
      req_id = req->get()->id();
      return receiver_.on_msg(
          req, ios_.wrap([&, cb, req_id](msg_ptr const& res,
                                         std::error_code const& ec) {
            cb(build_reply(req_id, res, ec));
          }));
    } catch (std::system_error const& e) {
      err = build_reply(req_id, nullptr, e.code());
    } catch (std::exception const& e) {
      err = make_unknown_error_msg(e.what());
    } catch (...) {
      err = make_unknown_error_msg("unknown");
    }
    err->get()->mutate_id(req_id);
    LOG(logging::error) << err->to_json();
    return cb(err);
  }

  static msg_ptr decode_msg(std::string const& req_buf, bool const binary) {
    if (binary) {
      return make_msg(req_buf.data(), req_buf.size());
    } else {
      return make_msg(req_buf, true);
    }
  }

  static std::string encode_msg(msg_ptr const& msg, bool const binary) {
    std::string b;
    if (binary) {
      b = std::string{reinterpret_cast<char const*>(msg->data()), msg->size()};
    } else {
      b = msg->to_json();
    }
    return b;
  }

  static msg_ptr build_reply(int const id, msg_ptr const& res,
                             std::error_code const& ec) {
    msg_ptr m = res;
    if (ec) {
      m = make_error_msg(ec);
    } else if (!res) {
      m = make_success_msg();
    }
    m->get()->mutate_id(id);
    return m;
  }

  void log_request(msg_ptr const& msg) {
    if (!logging_enabled_) {
      return;
    }

    try {
      log_file_ << msg->to_json(true) << std::endl;
    } catch (std::ios_base::failure const& e) {
      LOG(logging::error) << "could not log request: " << e.what();
      reset_logging(false);
    }
  }

  void reset_logging(bool rethrow) {
    try {
      log_file_ = std::ofstream{};
      log_file_.exceptions(std::ios_base::failbit | std::ios_base::badbit);
      log_file_.open(log_path_, std::ios_base::app);
      logging_enabled_ = true;
    } catch (std::ios_base::failure const& e) {
      LOG(logging::error) << "could not open logfile: " << e.what();
      if (rethrow) {
        throw;
      }
    }
  }

private:
#if defined(NET_TLS)
  ssl::context ctx_;
#endif
  boost::asio::io_service& ios_;
  receiver& receiver_;
  net::web_server server_;
  bool logging_enabled_{false};
  std::string log_path_;
  std::ofstream log_file_;
  std::string static_file_path_;
  bool serve_static_files_{false};
};

web_server::web_server(boost::asio::io_service& ios, receiver& recvr)
    : impl_(new impl(ios, recvr)) {}

web_server::~web_server() = default;

void web_server::listen(std::string const& host, std::string const& port,
#if defined(NET_TLS)
                        std::string const& cert_path,
                        std::string const& priv_key_path,
                        std::string const& dh_path,
#endif
                        std::string const& log_path,
                        std::string const& static_path,
                        boost::system::error_code& ec) {
  impl_->listen(host, port,
#if defined(NET_TLS)
                cert_path, priv_key_path, dh_path,
#endif
                log_path, static_path, ec);
}

void web_server::stop() { impl_->stop(); }

}  // namespace motis::launcher
