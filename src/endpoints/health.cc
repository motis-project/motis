#include "motis/endpoints/health.h"

#include <algorithm>

#include "net/bad_request_exception.h"

#include "utl/verify.h"

namespace motis::ep {

std::string_view health::operator()(boost::urls::url_view const& url) const {
  utl::verify(metrics_ != nullptr, "Not running");

  if (!config_.requires_rt_timetable_updates()) {
    return std::string_view("Running, RT not enabled");
  }

  auto const& families = metrics_->registry_.Collect();
  auto it = std::find_if(families.begin(), families.end(), [](auto const& v) {
    return v.name == "nigiri_gtfsrt_last_update_timestamp_seconds";
  });

  if (it != families.end()) {
    auto const& datasets = config_.timetable_->datasets_;
    auto n_expected = std::count_if(
        datasets.begin(), datasets.end(),
        [](auto const& pair) { return pair.second.rt_.has_value(); });
    auto const& metrics = (*it).metric;
    auto n_actual = std::count_if(metrics.begin(), metrics.end(),
                                  [](auto const& m) { return m.gauge.value > 0.0; });

    if (n_actual >= n_expected) {
      return std::string_view{"Running, RT applied"};
    }
  }

  throw utl::fail<net::bad_request_exception>("Running, RT not applied");
}

}  // namespace motis::ep