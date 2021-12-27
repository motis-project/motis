#pragma once

#include "net/http/client/client.h"
#include "net/http/client/https_client.h"

#include "motis/module/context/get_io_service.h"

namespace motis::module {

std::shared_ptr<ctx::future<ctx_data, std::pair<net::http::client::response,
                                                boost::system::error_code>>>
motis_http_req(net::http::client::request&& req) {
  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    auto f = std::make_shared<
        ctx::future<ctx_data, std::pair<net::http::client::response,
                                        boost::system::error_code>>>(
        ctx::op_id{});
    boost::asio::io_service ios;
    net::http::client::make_https(ios, req.address)
        ->query(req, [&](std::shared_ptr<net::ssl> const&,
                         net::http::client::response&& res,
                         boost::system::error_code ec) {
          f->set(std::pair{std::move(res), std::move(ec)});
        });
    ios.run();
    return f;
  } else {
    auto& ios = get_io_service();

    auto const op = ctx::current_op<ctx_data>();
    auto id = ctx::op_id(CTX_LOCATION);
    id.parent_index = op->id_.index;

    auto f = std::make_shared<
        ctx::future<ctx_data, std::pair<net::http::client::response,
                                        boost::system::error_code>>>(id);
    net::http::client::make_https(ios, req.address)
        ->query(req, [&](std::shared_ptr<net::ssl> const&,
                         net::http::client::response&& res,
                         boost::system::error_code ec) {
          f->set(std::pair{std::move(res), std::move(ec)});
        });

    return f;
  }
}

}  // namespace motis::module
