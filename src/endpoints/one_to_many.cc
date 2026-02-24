#include "motis/endpoints/one_to_many.h"

namespace motis::ep {

api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  auto const max_many = config_.get_limits().onetomany_max_many_;
  auto const max_travel_time_limit =
      config_.get_limits().street_routing_max_direct_seconds_;
  return one_to_many_handle_request(query, w_, l_, elevations_, max_many,
                                    max_travel_time_limit);
}

}  // namespace motis::ep
