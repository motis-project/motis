#include "motis/module/context/motis_http_req.h"

#include "boost/algorithm/string/predicate.hpp"

#include "net/http/client/http_client.h"
#include "net/http/client/https_client.h"

#include "motis/module/context/get_io_service.h"

namespace motis::module {

std::shared_ptr<ctx::future<ctx_data, net::http::client::response>>
motis_http_req_impl(char const* src_location, net::http::client::request req) {
  auto const make_http_cb =
      [](std::shared_ptr<
          ctx::future<ctx_data, net::http::client::response>> const& f) {
        return [f](auto const&, net::http::client::response&& res,
                   boost::system::error_code ec) {
          if (ec.failed()) {
            try {
              throw std::system_error{ec};
            } catch (...) {
              f->set(std::current_exception());
            }
          } else {
            if (auto const it = res.headers.find("location");
                it != end(res.headers)) {
            }
            f->set(std::move(res));
          }
        };
      };

  auto const make_http_req = [&](boost::asio::io_context& ios,
                                 ctx::op_id const& id) {
    auto f =
        std::make_shared<ctx::future<ctx_data, net::http::client::response>>(
            id);
    if (boost::algorithm::starts_with(req.address.port(), "https") ||
        req.address.port() == "443") {
      make_https(ios, req.address)->query(req, make_http_cb(f));
    } else if (boost::algorithm::starts_with(req.address.port(), "http") ||
               req.address.port() == "80") {
      make_http(ios, req.address)->query(req, make_http_cb(f));
    } else {
      throw utl::fail("unexpected port {} (not https or http)",
                      req.address.port());
    }
    return f;
  };

  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    boost::asio::io_service ios;
    auto const f = make_http_req(ios, ctx::op_id{});
    ios.run();
    return f;
  } else {
    auto const op = ctx::current_op<ctx_data>();
    auto id = ctx::op_id(src_location);
    id.parent_index = op->id_.index;
    return make_http_req(get_io_service(), id);
  }
}

}  // namespace motis::module
