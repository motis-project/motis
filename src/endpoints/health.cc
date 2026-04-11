#include "motis/endpoints/health.h"

namespace motis::ep {

health::response_t health::operator()(boost::urls::url_view const&) const {
  using status = boost::beast::http::status;

  auto rt_updated = metrics_->last_update_rt_.Value() > 0.0;
  auto gbfs_updated = metrics_->last_update_gbfs_.Value() > 0.0;
  auto has_elevator_feeds =
      config_.has_elevators() && config_.get_elevators()->url_.has_value();

  auto const content =
      response_t::second_type{.rt_ = rt_updated && config_.has_rt_feeds(),
                              .elevators_ = rt_updated && has_elevator_feeds,
                              .gbfs_ = gbfs_updated};

  if ((!config_.has_gbfs_feeds() || gbfs_updated) &&
      ((!config_.has_rt_feeds() && !has_elevator_feeds) || rt_updated)) {
    return {status::ok, content};
  }

  return {status::bad_request, content};
}

}  // namespace motis::ep