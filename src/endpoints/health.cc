#include "motis/endpoints/health.h"

#include "boost/json.hpp"

#include "net/web_server/responses.h"

#include "utl/verify.h"

#include "motis-api/motis-api.h"

namespace motis::ep {

net::reply health::operator()(net::route_request const& req, bool) const {
  utl::verify(metrics_ != nullptr, "Not running");

  bool rt_updated = metrics_->last_update_rt_.Value() > 0.0;
  bool gbfs_updated = metrics_->last_update_gbfs_.Value() > 0.0;
  bool has_elevator_feeds = config_.has_elevators() && config_.get_elevators()->url_.has_value();

  api::HealthResponse content = {
      .rt_ = rt_updated && config_.has_rt_feeds(),
      .elevators_ = rt_updated && has_elevator_feeds,
      .gbfs_ = gbfs_updated};

  if ((!config_.has_gbfs_feeds() || gbfs_updated) &&
      ((!config_.has_rt_feeds() && !has_elevator_feeds) || rt_updated)) {
    return string_response(
        req, boost::json::serialize(boost::json::value_from(content)),
        boost::beast::http::status::ok, "application/json");
  }

  return string_response(
      req, boost::json::serialize(boost::json::value_from(content)),
      boost::beast::http::status::bad_request, "application/json");
}

}  // namespace motis::ep