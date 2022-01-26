#include "motis/gbfs/gbfs.h"

#include "boost/algorithm/string/predicate.hpp"

#include "net/http/client/http_client.h"
#include "net/http/client/https_client.h"

#include "geo/point_rtree.h"

#include "motis/module/event_collector.h"
#include "motis/gbfs/free_bike.h"
#include "motis/gbfs/station.h"
#include "motis/gbfs/system_status.h"

using namespace motis::module;
using net::http::client::make_http;
using net::http::client::make_https;
using net::http::client::request;
using net::http::client::response;

namespace motis::gbfs {

struct positions {
  std::vector<station> stations_;
  std::vector<free_bike> bikes_;
  geo::point_rtree stations_rtree_;
  geo::point_rtree bikes_rtree_;
};

template <typename Fn>
void http_req(boost::asio::io_service& ios, std::string const& url_str,
              Fn&& cb) {
  auto http_cb = [cb = std::forward<Fn>(cb)](
                     auto const&, response const& res,
                     boost::system::error_code const& ec) {
    if (ec.failed()) {
      return cb("", ec.message());
    } else {
      return cb(res.body, "");
    }
  };

  auto req = request{url_str};
  if (boost::algorithm::starts_with(url_str, "https")) {
    make_https(ios, req.address)->query(req, std::move(http_cb));
  } else if (boost::algorithm::starts_with(url_str, "http")) {
    make_http(ios, req.address)->query(req, std::move(http_cb));
  } else {
    throw utl::fail("unexpected URL {} (not https or http)", url_str);
  }
}

struct gbfs::impl {
  explicit impl(config const& c) : config_{c} {
    boost::asio::io_service ios;
    std::vector<std::variant<std::string /* error */, std::vector<urls>>> urls;
    for (auto const& url : c.urls_) {
      http_req(ios, url,
               [&](std::string const& response, std::string const& err) {
                 err.empty() ? urls.emplace_back(read_system_status(response))
                             : urls.emplace_back(err);
               });
    }
  }

  msg_ptr route(schedule const&, msg_ptr const&) { return {}; }

  config const& config_;
};

gbfs::gbfs() : module("RIS", "gbfs") {
  param(config_.update_interval_minutes_, "update_interval",
        "update interval in minutes");
  param(config_.urls_, "urls", "URLs to fetch data from");
}

gbfs::~gbfs() = default;

void gbfs::import(import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "parking", reg,
      [this](event_collector::dependencies_map_t const&,
             event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("OSRM", [](msg_ptr const& msg) {
        using import::OSRMEvent;
        return msg->get()->content_type() == MsgContent_OSRMEvent &&
               motis_content(OSRMEvent, msg)->profile()->str() == "bike";
      });
}

void gbfs::init(motis::module::registry& r) {
  impl_ = std::make_unique<impl>(config_);
  r.register_op("/gbfs/route",
                [&](msg_ptr const& m) { return impl_->route(get_sched(), m); });
}

}  // namespace motis::gbfs
