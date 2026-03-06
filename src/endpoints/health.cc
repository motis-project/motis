#include "motis/endpoints/health.h"

#include "boost/beast/status.hpp"

namespace motis::ep {

net::reply health::operator()(boost::urls::url_view const& req) const {
  using status = boost::beast::http::status;

  return net::web_server::empty_res_t{
    (metrics_->total_trips_with_realtime_count_.Value() < 1.0) ? status::internal_server_error : status::ok,
    req.version()
  };
}

}  // namespace motis::ep