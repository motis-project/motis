#include "motis/vdv_rt/subscribe.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"

#include "motis/config.h"
#include "motis/http_req.h"
#include "motis/vdv_rt/connection.h"
#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {

std::string unsubscribe_body(config const& c) {
  auto doc = make_xml_doc();
  add_subscription_node(doc, c.vdv_rt_->client_name_)
      .append_child("AboLoeschenAlle")
      .append_child(pugi::node_pcdata)
      .set_value("true");
  return xml_to_str(doc);
}

void subscribe(boost::asio::io_context& ioc,
               config const& c,
               vdv_rt::connection& con) {
  boost::asio::co_spawn(
      ioc,
      [&c, &con]() -> boost::asio::awaitable<void> {
        auto executor = co_await boost::asio::this_coro::executor;
        auto timer = boost::asio::steady_timer{executor};
        auto ec = boost::system::error_code{};
        while (true) {
          auto const start = std::chrono::steady_clock::now();

          // unsubscribe
          co_await boost::asio::co_spawn(
              executor, [&c, &con]() -> boost::asio::awaitable<void> {
                try {
                  auto const res = co_await http_POST(
                      boost::urls::url{con.subscription_addr_},
                      vdv_rt::kHeaders, unsubscribe_body(c),
                      std::chrono::seconds{c.vdv_rt_->timeout_});
                  if (res.result_int() != 200U) {
                    fmt::println("[vdv_rt] unsubscribe failed: {}",
                                 get_http_body(res));
                  }
                } catch (std::exception const& e) {
                  fmt::println("[vdv_rt] unsubscribe failed: {}", e.what());
                  co_return;
                }
              });

          // subscribe

          timer.expires_at(
              start + std::chrono::seconds{c.vdv_rt_->subscription_renewal_});
          co_await timer.async_wait(
              boost::asio::redirect_error(boost::asio::use_awaitable, ec));
          if (ec == boost::asio::error::operation_aborted) {
            co_return;
          }
        }
      },
      boost::asio::detached);
}

}  // namespace motis::vdv_rt