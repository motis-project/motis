#include "motis/endpoints/health.h"

#include <algorithm>

#include "net/bad_request_exception.h"

#include "utl/verify.h"

namespace motis::ep {

std::string_view health::operator()(boost::urls::url_view const& url) const {
  utl::verify(metrics_ != nullptr, "Not running");

  if (!config_.has_rt_feeds()) {
    return std::string_view("Running, RT not enabled");
  }

  if (metrics_->rt_last_update_.Value() > 0.0) {
    return std::string_view("Running, RT enabled");
  }

  throw utl::fail<net::bad_request_exception>("RT not applied");
}

}  // namespace motis::ep