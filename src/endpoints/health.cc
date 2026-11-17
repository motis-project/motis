#include "motis/endpoints/health.h"

#include "motis-api/motis-api.h"
#include "motis/health.h"

namespace motis::ep {

health::response_t health::operator()(boost::urls::url_view const&) const {
  using status = boost::beast::http::status;

  auto const health = api::HealthResponse{
      .rt_ = rt_updated(*metrics_) && config_.has_rt_feeds(),
      .gbfs_ = gbfs_updated(*metrics_)};

  return {is_healthy(config_, *metrics_) ? status::ok : status::bad_request,
          health};
}

}  // namespace motis::ep