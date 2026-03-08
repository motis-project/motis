#include "motis/endpoints/health.h"

#include <algorithm>

#include "net/bad_request_exception.h"

#include "utl/verify.h"

namespace motis::ep {

std::string_view health::operator()(boost::urls::url_view const& url) const {
  utl::verify(metrics_ != nullptr, "Not running");
  config_.verify();

  if (!config_.requires_rt_timetable_updates()) {
    return std::string_view("Running, RT not enabled");
  }

  auto const& families = metrics_->registry_.Collect();
  auto it = std::find_if(families.begin(), families.end(), [](auto const& v) {
    return v.name == "nigiri_gtfsrt_last_update_timestamp_seconds";
  });

  if (it != families.end() && (*it).metric[0].gauge.value > 0.0) {
    return std::string_view{"Running, RT applied"};
  }

  throw utl::fail<net::bad_request_exception>("Running, RT not applied");
}

}  // namespace motis::ep