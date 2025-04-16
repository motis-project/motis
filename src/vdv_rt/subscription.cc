#include "motis/vdv_rt/subscription.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/http_req.h"
#include "motis/vdv_rt/connection.h"
#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {

static auto const kHeaders = std::map<std::string, std::string>{
    {"Content-Type", "text/xml"}, {"Accept", "text/xml"}};

pugi::xml_node add_sub_req_node(pugi::xml_node& node,
                                std::string const& sender) {
  auto sub_req_node = node.append_child("AboAnfrage");
  sub_req_node.append_attribute("Sender") = sender.c_str();
  sub_req_node.append_attribute("Zst") = timestamp(now()).c_str();
  return sub_req_node;
}

std::string unsubscribe_body(connection const& vdv_rt) {
  auto doc = make_xml_doc();
  add_sub_req_node(doc, vdv_rt.cfg_.client_name_)
      .append_child("AboLoeschenAlle")
      .append_child(pugi::node_pcdata)
      .set_value("true");
  return xml_to_str(doc);
}

std::string subscribe_body(config const& c, connection const& vdv_rt) {
  auto doc = make_xml_doc();
  auto sub_req_node = add_sub_req_node(doc, vdv_rt.cfg_.client_name_);
  auto sub_node = sub_req_node.append_child("AboAUS");
  sub_node.append_attribute("AboID") = "1";
  sub_node.append_attribute("VerfallZst") =
      timestamp(
          now() +
          std::chrono::seconds{c.timetable_->vdv_rt_subscription_duration_})
          .c_str();
  auto hysteresis_node = sub_node.append_child("Hysterese");
  hysteresis_node.append_child(pugi::node_pcdata)
      .set_value(std::to_string(vdv_rt.cfg_.hysteresis_).c_str());
  auto lookahead_node = sub_node.append_child("Vorschauzeit");
  lookahead_node.append_child(pugi::node_pcdata)
      .set_value(
          std::to_string(std::chrono::round<std::chrono::minutes>(
                             std::chrono::seconds{
                                 c.timetable_->vdv_rt_subscription_duration_})
                             .count())
              .c_str());
  return xml_to_str(doc);
}

boost::asio::awaitable<void> unsubscribe(boost::asio::io_context& ioc,
                                         config const& c,
                                         data& d) {
  co_await boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> boost::asio::awaitable<void> {
        auto executor = co_await boost::asio::this_coro::executor;
        auto awaitables = utl::to_vec(*d.vdv_rt_, [&](auto&& con) {
          return boost::asio::co_spawn(
              executor,
              [&c, &con]() -> boost::asio::awaitable<void> {
                try {
                  auto const res = co_await http_POST(
                      boost::urls::url{con.subscription_addr_}, kHeaders,
                      unsubscribe_body(con),
                      std::chrono::seconds{c.timetable_->http_timeout_});
                  if (res.result_int() != 200U) {
                    fmt::println("[vdv_rt] unsubscribe failed: {}",
                                 get_http_body(res));
                  }
                } catch (std::exception const& e) {
                  fmt::println("[vdv_rt] unsubscribe failed: {}", e.what());
                }
              },
              boost::asio::deferred);
        });
        co_await boost::asio::experimental::make_parallel_group(awaitables)
            .async_wait(boost::asio::experimental::wait_for_all(),
                        boost::asio::use_awaitable);
      },
      boost::asio::use_awaitable);
}

boost::asio::awaitable<void> subscribe(boost::asio::io_context& ioc,
                                       config const& c,
                                       data& d) {
  co_await boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> boost::asio::awaitable<void> {
        auto executor = co_await boost::asio::this_coro::executor;
        auto awaitables = utl::to_vec(*d.vdv_rt_, [&](auto&& con) {
          return boost::asio::co_spawn(
              executor,
              [&c, &con]() -> boost::asio::awaitable<void> {
                try {
                  auto const res = co_await http_POST(
                      boost::urls::url{con.subscription_addr_}, kHeaders,
                      subscribe_body(c, con),
                      std::chrono::seconds{c.timetable_->http_timeout_});
                  if (res.result_int() == 200U) {
                    con.start_ = now();
                  } else {
                    fmt::println("[vdv_rt] subscribe failed: {}",
                                 get_http_body(res));
                  }
                } catch (std::exception const& e) {
                  fmt::println("[vdv_rt] subscribe failed: {}", e.what());
                }
              },
              boost::asio::deferred);
        });
        co_await boost::asio::experimental::make_parallel_group(awaitables)
            .async_wait(boost::asio::experimental::wait_for_all(),
                        boost::asio::use_awaitable);
      },
      boost::asio::use_awaitable);
}

void subscription(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d, &ioc]() -> boost::asio::awaitable<void> {
        auto executor = co_await boost::asio::this_coro::executor;
        auto timer = boost::asio::steady_timer{executor};
        auto ec = boost::system::error_code{};
        while (true) {
          auto const start = std::chrono::steady_clock::now();

          co_await unsubscribe(ioc, c, d);
          co_await subscribe(ioc, c, d);

          timer.expires_at(
              start +
              std::chrono::seconds{c.timetable_->vdv_rt_subscription_renewal_});
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