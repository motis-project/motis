#pragma once

#include <iostream>

#include "boost/asio/io_context.hpp"

#include "ctx/scheduler.h"

#include "motis/ctx_data.h"

namespace motis {

struct ctx_exec {
  ctx_exec(boost::asio::io_context& io, ctx::scheduler<ctx_data>& sched)
      : io_{io}, sched_{sched} {}

  void exec(auto&& f, net::web_server::http_res_cb_t cb) {
    sched_.post_void_io(
        ctx_data{},
        [&, f = std::move(f), cb = std::move(cb)]() mutable {
          try {
            auto res = std::make_shared<net::web_server::http_res_t>(f());
            io_.post([cb = std::move(cb), res = std::move(res)]() mutable {
              cb(std::move(*res));
            });
          } catch (...) {
            std::cerr << "UNEXPECTED EXCEPTION\n";

            auto str = net::web_server::string_res_t{
                boost::beast::http::status::internal_server_error, 11};
            str.body() = "error";
            str.prepare_payload();

            auto res = std::make_shared<net::web_server::http_res_t>(str);
            io_.post([cb = std::move(cb), res = std::move(res)]() mutable {
              cb(std::move(*res));
            });
          }
        },
        CTX_LOCATION);
  }

  boost::asio::io_context& io_;
  ctx::scheduler<ctx_data>& sched_;
};

}  // namespace motis