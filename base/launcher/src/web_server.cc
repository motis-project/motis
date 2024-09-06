#include "motis/launcher/web_server.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>

#include "boost/beast/version.hpp"
#include "boost/url/url_view.hpp"

#include "fmt/format.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/sdk/resource/semantic_conventions.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "net/web_server/content_encoding.h"
#include "net/web_server/responses.h"
#include "net/web_server/serve_static.h"
#include "net/web_server/web_server.h"

#include "motis/core/common/logging.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/client.h"
#include "motis/launcher/http_text_map_carrier.h"
#include "motis/launcher/load_server_certificate.h"

#if defined(NET_TLS)
namespace ssl = boost::asio::ssl;
#endif

namespace fs = std::filesystem;
using namespace motis::module;

namespace motis::launcher {

std::string encode_msg(msg_ptr const& msg, bool const binary,
                       json_format const jf = kDefaultOuputJsonFormat) {
  std::string b;
  if (binary) {
    b = std::string{reinterpret_cast<char const*>(msg->data()), msg->size()};
  } else {
    b = msg->to_json(jf);
  }
  return b;
}

msg_ptr build_reply(int const id, msg_ptr const& res,
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

std::pair<msg_ptr, json_format> decode_msg(
    std::string const& req_buf, bool const binary,
    std::string_view const target = std::string_view{}) {
  auto jf = kDefaultOuputJsonFormat;
  if (binary) {
    return {make_msg(req_buf.data(), req_buf.size()), jf};
  } else {
    auto const msg = make_msg(req_buf, jf, true, target);
    return {msg, jf};
  }
}

inline std::string_view to_sv(boost::beast::string_view const& bsv) {
  return {bsv.data(), bsv.size()};
}

struct ws_client : public client,
                   public std::enable_shared_from_this<ws_client> {
  ws_client(boost::asio::io_service& ios, net::ws_session_ptr session,
            bool binary)
      : ios_{ios}, session_{std::move(session)}, binary_{binary} {}

  ws_client(ws_client const&) = delete;
  ws_client(ws_client&&) = delete;
  ws_client& operator=(ws_client const&) = delete;
  ws_client& operator=(ws_client&&) = delete;

  ~ws_client() override = default;

  void set_on_msg_cb(
      std::function<void(msg_ptr const&, json_format)>&& cb) override {
    if (auto const lock = session_.lock(); lock) {
      lock->on_msg([this, cb = std::move(cb)](std::string const& req_buf,
                                              net::ws_msg_type const type) {
        if (!cb) {
          return;
        }

        msg_ptr err;
        int req_id = 0;

        try {
          auto const [req, jf] =
              decode_msg(req_buf, type == net::ws_msg_type::BINARY);
          req_id = req->get()->id();
          return cb(req, jf);
        } catch (std::system_error const& e) {
          err = build_reply(req_id, nullptr, e.code());
        } catch (std::exception const& e) {
          err = make_unknown_error_msg(e.what());
        } catch (...) {
          err = make_unknown_error_msg("unknown");
        }
        err->get()->mutate_id(req_id);

        if (auto const lock = session_.lock(); lock) {
          lock->send(encode_msg(err, type == net::ws_msg_type::BINARY), type,
                     [](boost::system::error_code, std::size_t) {});
        }
      });
    }
  }

  void set_on_close_cb(std::function<void()>&& cb) override {
    auto const lock = session_.lock();
    if (lock) {
      lock->on_close([s = shared_from_this(), cb = std::move(cb)]() {
        if (cb) {
          cb();
        }
      });
    }
  }

  void send(msg_ptr const& m, json_format const jf) override {
    ios_.post([self = shared_from_this(), m, jf]() {
      if (auto s = self->session_.lock()) {
        s->send(
            encode_msg(m, self->binary_, jf),
            self->binary_ ? net::ws_msg_type::BINARY : net::ws_msg_type::TEXT,
            [](boost::system::error_code, size_t) {});
      }
    });
  }

  boost::asio::io_service& ios_;
  net::ws_session_ptr session_;
  bool binary_;
};

struct web_server::impl {
#if defined(NET_TLS)
  impl(boost::asio::io_service& ios, controller& ctr)
      : ctx_{ssl::context::tlsv12},
        ios_{ios},
        receiver_{ctr},
        server_{ios, ctx_} {}
#else
  impl(boost::asio::io_service& ios, controller& ctr)
      : ios_{ios}, receiver_{ctr}, server_{ios} {}
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
    server_.on_ws_open([this](net::ws_session_ptr const& s,
                              std::string const& target,
                              bool /* ssl */) { on_ws_open(s, target); });
    server_.on_upgrade_ok([this](net::web_server::http_req_t const& req) {
      return req.target() == "/" ||
             receiver_.connect_ok(std::string{req.target()});
    });
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
      std::cerr << "Static file directory not found: " << e.what() << '\n';
    }
    if (serve_static_files_) {
      std::cout << "Serving static files from " << static_file_path_ << '\n';
    } else {
      std::cout << "Not serving static files" << '\n';
    }
    if (!ec) {
      server_.run();
    }
  }

  void stop() { server_.stop(); }

  void on_http_request(net::web_server::http_req_t const& req,
                       net::web_server::http_res_cb_t const& cb) {
    using namespace boost::beast::http;
    using namespace opentelemetry::sdk::resource;

    auto otel_propagator = opentelemetry::context::propagation::
        GlobalTextMapPropagator::GetGlobalPropagator();
    auto carrier = http_text_map_carrier{req};
    auto current_ctx = opentelemetry::context::RuntimeContext::GetCurrent();
    auto new_ctx = otel_propagator->Extract(carrier, current_ctx);

    auto const url = boost::urls::url_view{req.target()};

    auto span = motis_tracer->StartSpan(
        req.method_string(),
        {
            {SemanticConventions::kHttpRequestMethod, req.method_string()},
            {SemanticConventions::kUrlPath, url.path()},
            {SemanticConventions::kUrlQuery, url.query()},
            {SemanticConventions::kUrlScheme, "http"},
        },
        opentelemetry::trace::StartSpanOptions{
            .parent = opentelemetry::trace::GetSpan(new_ctx)->GetContext(),
            .kind = opentelemetry::trace::SpanKind::kServer});
    auto scope = opentelemetry::trace::Scope{span};

    if (auto const user_agent = req[field::user_agent]; !user_agent.empty()) {
      span->SetAttribute(SemanticConventions::kUserAgentOriginal, user_agent);
    }

    auto const build_response = [req, span](msg_ptr const& response,
                                            std::optional<json_format> jf =
                                                std::nullopt) {
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

      span->SetAttribute(SemanticConventions::kHttpResponseStatusCode,
                         res.result_int());
      if (res.result() == status::internal_server_error) {
        span->SetStatus(opentelemetry::trace::StatusCode::kError);
        span->SetAttribute(SemanticConventions::kErrorType,
                           fmt::to_string(res.result_int()));
      }

      std::string content;
      auto has_already_content_encoding = false;
      if (response != nullptr &&
          response->get()->content_type() == MsgContent_HTTPResponse) {
        auto const http_res = motis_content(HTTPResponse, response);
        res.result(http_res->status() == HTTPStatus_OK
                       ? http_res->content()->size() != 0 ? status::ok
                                                          : status::no_content
                       : status::internal_server_error);

        for (auto const& h : *http_res->headers()) {
          res.set(h->name()->str(), h->value()->str());
        }

        has_already_content_encoding =
            utl::find_if(*http_res->headers(), [](HTTPHeader const* hdr) {
              return boost::beast::iequals("content-encoding",
                                           hdr->name()->view());
            }) != std::end(*http_res->headers());
        content = http_res->content()->str();
      } else {
        res.set(field::content_type, "application/json");
        if (response != nullptr) {
          content = response->to_json(jf.value_or(kDefaultOuputJsonFormat));
        }
      }

      if (has_already_content_encoding) {
        res.body() = content;
      } else {
        net::set_response_body(res, req, content);
      }
      if (!content.empty()) {
        res.prepare_payload();
      }
      return res;
    };

    auto const res_cb = [cb, build_response](msg_ptr const& response,
                                             std::optional<json_format> jf) {
      cb(build_response(response, jf));
    };

    std::string req_msg;
    switch (req.method()) {
      case verb::options: return cb(build_response(nullptr));
      case verb::post: {
        auto const content_type = req[field::content_type];
        if (!content_type.empty() &&
            !boost::beast::iequals(content_type, "application/json")) {
          return on_generic_req(req, res_cb);
        }
        req_msg = req.body();
        span->SetAttribute(SemanticConventions::kHttpRequestBodySize,
                           req_msg.size());
        span->SetAttribute("motis.http.request.body", req_msg);
        if (req_msg.empty()) {
          req_msg = make_no_msg(std::string{req.target()})
                        ->to_json(kDefaultOuputJsonFormat);
        }
        break;
      }
      case verb::head:
      case verb::get:
        if (serve_static_files_ &&
            net::serve_static_file(static_file_path_, req, cb)) {
          span->UpdateName(fmt::format("{} /{{static}}", req.method_string()));
          span->SetAttribute(
              opentelemetry::sdk::resource::SemanticConventions::kHttpRoute,
              "/{static}");
          return;
        } else {
          req_msg = make_no_msg(std::string{req.target()})
                        ->to_json(kDefaultOuputJsonFormat);
          break;
        }
      default:
        return cb(build_response(make_error_msg(
            std::make_error_code(std::errc::operation_not_supported))));
    }

    return on_msg_req(req_msg, false, to_sv(req.method_string()),
                      to_sv(req.target()), res_cb);
  }

  void on_ws_open(net::ws_session_ptr session, std::string const& target) {
    LOG(logging::info) << "ws connection to \"" << target << "\"";
    if (target != "/") {
      auto const c =
          std::make_shared<ws_client>(ios_, std::move(session), false);
      auto const s = c->session_.lock();
      if (s) {
        s->on_close([c]() {});
        receiver_.on_connect(target, c);
      }
    }
  }

  void on_ws_msg(net::ws_session_ptr const& session, std::string const& msg,
                 net::ws_msg_type type) {
    auto const is_binary = type == net::ws_msg_type::BINARY;
    auto span = motis_tracer->StartSpan(
        "WebSocket",
        {
            {"format", is_binary ? "binary" : "text"},
        },
        opentelemetry::trace::StartSpanOptions{
            .kind = opentelemetry::trace::SpanKind::kServer});
    auto scope = opentelemetry::trace::Scope{span};
    return on_msg_req(
        msg, is_binary, "WebSocket", {},
        [session, type, is_binary](msg_ptr const& response,
                                   std::optional<json_format> jf) {
          if (auto s = session.lock()) {
            s->send(encode_msg(response, is_binary,
                               jf.value_or(kDefaultOuputJsonFormat)),
                    type, [](boost::system::error_code, size_t) {});
          }
        });
  }

  void on_msg_req(
      std::string const& request, bool binary, std::string_view const method,
      std::string_view const target,
      std::function<void(msg_ptr const&, std::optional<json_format>)> const& cb,
      std::optional<json_format> jf = std::nullopt) {
    msg_ptr err;
    int req_id = 0;
    try {
      auto const [req, detected_jf] = decode_msg(request, binary, target);
      if (!jf) {
        jf = detected_jf;
      }

      log_request(req);
      auto const* msg = req->get();
      req_id = msg->id();

      auto span = motis_tracer->GetCurrentSpan();
      auto const op_name =
          receiver_.get_operation_name(msg->destination()->target()->str());
      if (op_name) {
        span->UpdateName(fmt::format("{} {}", method, *op_name));
        span->SetAttribute(
            opentelemetry::sdk::resource::SemanticConventions::kHttpRoute,
            *op_name);
      }
      span->SetAttribute("motis.message.id", req_id);
      span->SetAttribute("motis.message.target",
                         msg->destination()->target()->view());
      span->SetAttribute("motis.message.type",
                         EnumNameMsgContent(msg->content_type()));

      return receiver_.on_msg(
          req, ios_.wrap([cb, req_id, jf](msg_ptr const& res,
                                          std::error_code const& ec) {
            cb(build_reply(req_id, res, ec), jf);
          }));
    } catch (std::system_error const& e) {
      err = build_reply(req_id, nullptr, e.code());
    } catch (std::exception const& e) {
      err = make_unknown_error_msg(e.what());
    } catch (...) {
      err = make_unknown_error_msg("unknown");
    }
    err->get()->mutate_id(req_id);
    return cb(err, jf);
  }

  void on_generic_req(
      net::web_server::http_req_t const& req,
      std::function<void(msg_ptr const&, std::optional<json_format>)> const& cb,
      std::optional<json_format> jf = std::nullopt) {
    using namespace boost::beast::http;

    auto const req_id = 1;
    msg_ptr err;
    try {
      message_creator mc;
      mc.create_and_finish(
          MsgContent_HTTPRequest,
          CreateHTTPRequest(
              mc, HTTPMethod_POST, mc.CreateString(to_sv(req.target())),
              mc.CreateVector(
                  utl::to_vec(req,
                              [&](auto const& f) {
                                return CreateHTTPHeader(
                                    mc, mc.CreateString(to_sv(f.name_string())),
                                    mc.CreateString(to_sv(f.value())));
                              })),
              mc.CreateString(req.body()))
              .Union(),
          std::string{req.target()});
      auto const msg = make_msg(mc);
      return receiver_.on_msg(
          msg,
          ios_.wrap([&, cb](msg_ptr const& res, std::error_code const& ec) {
            cb(build_reply(req_id, res, ec), jf);
          }));
    } catch (std::system_error const& e) {
      err = build_reply(req_id, nullptr, e.code());
    } catch (std::exception const& e) {
      err = make_unknown_error_msg(e.what());
    } catch (...) {
      err = make_unknown_error_msg("unknown");
    }
    LOG(logging::error) << err->to_json();
    return cb(err, jf);
  }

  void log_request(msg_ptr const& msg) {
    if (!logging_enabled_) {
      return;
    }

    try {
      log_file_ << "[" << motis::logging::time() << "] "
                << msg->to_json(json_format::SINGLE_LINE) << '\n';
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
  controller& receiver_;
  net::web_server server_;
  bool logging_enabled_{false};
  std::string log_path_;
  std::ofstream log_file_;
  std::string static_file_path_;
  bool serve_static_files_{false};
};

web_server::web_server(boost::asio::io_service& ios, controller& ctr)
    : impl_(new impl(ios, ctr)) {}

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
